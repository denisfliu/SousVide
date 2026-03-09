"""
SplatNav-based recovery module.

When the falsification framework detects a failure, this module:
1. Takes the last safe state from the failure detector
2. Uses SplatPlan to compute a collision-free trajectory from that state to the goal
3. Converts the SplatPlan trajectory into FiGS-compatible state/control format
   so it can be used for additional corrective data generation

The recovery trajectory represents a "what should have happened" correction
that can be used for:
- Generating corrective training data (rollout the safe trajectory in FiGS)
- Analyzing failure modes (compare VLA trajectory vs. safe trajectory)
- Iterative policy improvement (DAgger-style)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sousvide.falsification.failure_detector import FailureRecord, StateSnapshot


@dataclass
class RecoveryConfig:
    """Configuration for the SplatNav recovery planner."""
    # robot_radius in Nerfstudio-internal (NS) frame units.
    # 1 NS unit ≈ 5.78 m MOCAP for the gate scenes (dp_scale≈0.112, s≈0.649).
    # 0.02 NS ≈ 0.116 m MOCAP — small enough to fit through the gate opening.
    robot_radius: float = 0.02
    vmax: float = 2.0
    amax: float = 3.0
    spline_degree: int = 6
    n_spline_sections: int = 10
    # Voxel bounds in NS-internal frame.  ±0.5 NS covers ≈ ±2.9 m MOCAP.
    env_lower_bound: List[float] = field(default_factory=lambda: [-0.5, -0.5, -0.5])
    env_upper_bound: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    voxel_resolution: int = 150
    gate_position: Optional[List[float]] = None
    gate_pass_radius_m: float = 0.25
    goal_tolerance_m: float = 0.20
    gate_approach_offset_m: float = 0.4


@dataclass
class RecoveryResult:
    """Result of a SplatNav recovery planning attempt."""
    feasible: bool
    safe_state: StateSnapshot
    goal_position: np.ndarray
    trajectory_positions: np.ndarray | None = None     # (N, 3)
    astar_path: np.ndarray | None = None               # (M, 3)
    polytopes: list | None = None
    planning_time_s: float = 0.0
    metadata: Dict = field(default_factory=dict)


class SplatNavRecovery:
    """
    Wraps the SplatNav planner for recovery trajectory generation.

    This class manages the lifecycle of loading the SplatNav components,
    performing coordinate transforms between FiGS (NED) and SplatNav
    (nerfstudio) frames, and converting planned paths back to FiGS format.
    """

    def __init__(
        self,
        gsplat_path: str | Path,
        config: RecoveryConfig | None = None,
        device: torch.device | None = None,
        coordinate_transform: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        gsplat_path : path
            Path to the nerfstudio config.yml for the Gaussian splat model.
        config : RecoveryConfig
        device : torch.device
        coordinate_transform : (4, 4) array, optional
            Transform from FiGS world frame (NED) to nerfstudio frame.
            If None, identity is used (assumes frames are aligned).
        """
        self.config = config or RecoveryConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gsplat_path = Path(gsplat_path)
        self.T_figs_to_nerf = (
            np.asarray(coordinate_transform, dtype=np.float64)
            if coordinate_transform is not None
            else np.eye(4)
        )
        self.T_nerf_to_figs = np.linalg.inv(self.T_figs_to_nerf)

        self._planner = None
        self._gsplat = None

    # ------------------------------------------------------------------
    # Lazy initialisation (heavy GPU load)
    # ------------------------------------------------------------------

    def _ensure_planner(self) -> None:
        """Load SplatNav components on first use."""
        if self._planner is not None:
            return

        from splat.splat_utils import GSplatLoader
        from splatplan.splatplan import SplatPlan
        from splatplan.spline_utils import SplinePlanner

        cfg = self.config

        self._gsplat = GSplatLoader(self.gsplat_path, self.device)

        robot_config = {
            "radius": cfg.robot_radius,
            "vmax": cfg.vmax,
            "amax": cfg.amax,
        }

        env_config = {
            "lower_bound": torch.tensor(cfg.env_lower_bound, device=self.device),
            "upper_bound": torch.tensor(cfg.env_upper_bound, device=self.device),
            "resolution": cfg.voxel_resolution,
        }

        spline_planner = SplinePlanner(
            spline_deg=cfg.spline_degree,
            N_sec=cfg.n_spline_sections,
            device=self.device,
        )

        self._planner = SplatPlan(
            self._gsplat, robot_config, env_config, spline_planner, self.device
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _figs_pos_to_nerf(self, pos_figs: np.ndarray) -> np.ndarray:
        """Convert a 3D position from FiGS NED frame to nerfstudio frame."""
        p_h = np.append(pos_figs, 1.0)
        return (self.T_figs_to_nerf @ p_h)[:3]

    def _nerf_pos_to_figs(self, pos_nerf: np.ndarray) -> np.ndarray:
        """Convert a 3D position from nerfstudio frame to FiGS NED frame."""
        p_h = np.append(pos_nerf, 1.0)
        return (self.T_nerf_to_figs @ p_h)[:3]

    def _validate_trajectory_positions(
        self,
        trajectory_positions: np.ndarray | None,
        goal_position_figs: np.ndarray,
    ) -> tuple[bool, Dict]:
        """Check whether a planned recovery path actually goes through the gate
        and terminates close enough to the goal."""
        if trajectory_positions is None or len(trajectory_positions) == 0:
            return False, {"reason": "empty_trajectory"}

        metadata: Dict[str, object] = {}
        passes_gate = True
        min_gate_distance = None
        gate_position = self.config.gate_position
        if gate_position is not None:
            gate = np.asarray(gate_position, dtype=float)
            dists = np.linalg.norm(trajectory_positions - gate[None, :], axis=1)
            min_gate_distance = float(np.min(dists))
            passes_gate = bool(min_gate_distance <= float(self.config.gate_pass_radius_m))
            metadata["passes_gate"] = passes_gate
            metadata["min_gate_distance_m"] = min_gate_distance
            metadata["gate_pass_radius_m"] = float(self.config.gate_pass_radius_m)

        goal = np.asarray(goal_position_figs, dtype=float)
        final_goal_distance = float(np.linalg.norm(trajectory_positions[-1] - goal))
        reaches_goal = bool(final_goal_distance <= float(self.config.goal_tolerance_m))
        metadata["reaches_goal"] = reaches_goal
        metadata["final_goal_distance_m"] = final_goal_distance
        metadata["goal_tolerance_m"] = float(self.config.goal_tolerance_m)

        valid = passes_gate and reaches_goal
        if not valid:
            reasons = []
            if not passes_gate:
                reasons.append("misses_gate")
            if not reaches_goal:
                reasons.append("misses_goal")
            metadata["validation_error"] = ",".join(reasons)
        return valid, metadata

    def _validate_recovery_trajectory(
        self,
        failure_record: FailureRecord,
        recovery_positions: np.ndarray | None,
        goal_position_figs: np.ndarray,
    ) -> tuple[bool, Dict]:
        """Validate the stitched trajectory from rollout start to goal.

        If the failure record includes the rollout prefix, we require the
        concatenated prefix+recovery path to satisfy the gate/goal checks.
        """
        stitched = recovery_positions
        prefix_positions = []
        if failure_record.trajectory_up_to_failure:
            for snap in failure_record.trajectory_up_to_failure:
                if snap.step_index <= failure_record.last_safe_step:
                    prefix_positions.append(np.asarray(snap.state[:3], dtype=float))
            if prefix_positions:
                prefix_arr = np.asarray(prefix_positions, dtype=float)
                if recovery_positions is None or len(recovery_positions) == 0:
                    stitched = prefix_arr
                else:
                    if np.allclose(prefix_arr[-1], recovery_positions[0]):
                        stitched = np.vstack([prefix_arr[:-1], recovery_positions])
                    else:
                        stitched = np.vstack([prefix_arr, recovery_positions])

        valid, metadata = self._validate_trajectory_positions(stitched, goal_position_figs)
        metadata["stitched_trajectory"] = bool(prefix_positions)
        metadata["prefix_points"] = int(len(prefix_positions))
        metadata["recovery_points"] = int(0 if recovery_positions is None else len(recovery_positions))
        return valid, metadata

    # ------------------------------------------------------------------
    # Recovery planning
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Min-snap + SplatNav-validation helpers
    # ------------------------------------------------------------------

    def _min_snap_positions(
        self,
        waypoints_figs: List[np.ndarray],
        total_time: float = 5.0,
        hz: int = 10,
    ) -> np.ndarray:
        """Generate a min-snap trajectory through ``waypoints_figs`` (FiGS NED).

        Returns an (N, 3) array of position samples at ``hz`` Hz.
        Uses FiGS's ``MinTimeSnap`` and ``TsFO_to_tXU`` directly.
        """
        from figs.tsplines.min_time_snap import MinTimeSnap
        import figs.utilities.transform_helper as th

        n = len(waypoints_figs)
        keyframes: Dict = {}
        for i, pos in enumerate(waypoints_figs):
            t = (i / max(n - 1, 1)) * total_time
            name = f"wp_{i:03d}"
            if i == 0 or i == n - 1:
                # Fixed position and zero velocity at endpoints
                keyframes[name] = {"t": t, "fo": [
                    [float(pos[0]), 0.0],
                    [float(pos[1]), 0.0],
                    [float(pos[2]), 0.0],
                    [0.0, 0.0],
                ]}
            else:
                # Position constrained, derivatives free
                keyframes[name] = {"t": t, "fo": [
                    [float(pos[0]), None, None, None],
                    [float(pos[1]), None, None, None],
                    [float(pos[2]), None, None, None],
                    [0.0, None, None, None],
                ]}

        wps_cfg = {"Nco": 6, "keyframes": keyframes}
        mts = MinTimeSnap(wps_cfg, hz, kT=10.0, use_l2_time=False)
        Tsd, FOd = mts.get_desired_trajectory()
        # mass=1.0, kt=7.0 (defaults) — we only need positions here
        tXUd = th.TsFO_to_tXU(Tsd, FOd, 1.0, 7.0, None)
        return tXUd[:, 1:4]   # (N, 3) positions in FiGS NED

    def _trajectory_collision_free(self, positions_figs: np.ndarray, stride: int = 3) -> bool:
        """Return True iff no sampled position falls in the SplatNav voxel's
        non-navigable (obstacle-inflated) cells.

        Points are converted from FiGS NED to Nerfstudio before querying.
        """
        voxel = self._planner.gsplat_voxel
        for pos in positions_figs[::stride]:
            p_nerf = torch.tensor(
                self._figs_pos_to_nerf(pos), dtype=torch.float32, device=self.device
            )
            idx = voxel.get_indices(p_nerf)
            if voxel.non_navigable_grid[idx[0], idx[1], idx[2]]:
                return False
        return True

    # ------------------------------------------------------------------
    # plan_via_gate
    # ------------------------------------------------------------------

    def plan_via_gate(
        self,
        start_position_figs: np.ndarray,
        gate_position_figs: np.ndarray,
        goal_position_figs: np.ndarray,
        start_state_figs: np.ndarray | None = None,
    ) -> RecoveryResult:
        """Plan a safe path: start → gate_opening → goal using min-snap + collision validation.

        Strategy
        --------
        1. Place a ``before_gate`` waypoint (approach side) and ``after_gate``
           waypoint (exit side) at ±``gate_approach_offset_m`` along the
           start→gate direction.
        2. Generate a min-snap trajectory through
           ``[start, before_gate, gate, after_gate, goal]`` using FiGS's
           ``MinTimeSnap`` planner.  This guarantees smoothness and dynamic
           feasibility.
        3. Query every sampled position against the SplatNav collision voxel.
           If collision-free → accept.
        4. If in collision, perturb ``before_gate`` and ``after_gate``
           perpendicular to the approach direction (a grid of small lateral +
           vertical offsets) and repeat.  Up to ``max_perturb_attempts``
           candidates are tried.

        Parameters
        ----------
        start_position_figs : array (3,)
            Start position in FiGS (NED) coordinates.
        gate_position_figs : array (3,)
            Perturbed gate centre in FiGS (NED) coordinates.
        goal_position_figs : array (3,)
            Goal position in FiGS (NED) coordinates.
        start_state_figs : array (10,), optional
            Full FiGS state at the start (position + velocity + quaternion).
        """
        import time as _time

        self._ensure_planner()

        start  = np.asarray(start_position_figs,  dtype=float)
        gate   = np.asarray(gate_position_figs,   dtype=float)
        goal   = np.asarray(goal_position_figs,   dtype=float)
        offset = float(self.config.gate_approach_offset_m)

        # ---- Approach and lateral directions --------------------------------
        approach = gate - start
        d = np.linalg.norm(approach)
        approach = approach / max(d, 1e-6)

        # Two perpendicular directions in 3D (used for lateral perturbations)
        up_world = np.array([0., 0., -1.])  # "up" in NED (z is down, so −z is up)
        lateral = np.cross(approach, up_world)
        lat_norm = np.linalg.norm(lateral)
        if lat_norm < 1e-6:
            # approach is vertical — choose any perpendicular
            lateral = np.array([1., 0., 0.])
        else:
            lateral = lateral / lat_norm
        vertical = np.cross(approach, lateral)  # stays unit-length

        # ---- Build dummy start snapshot for RecoveryResult ------------------
        if start_state_figs is not None and len(start_state_figs) == 10:
            snap_state = np.array(start_state_figs, dtype=float)
        else:
            snap_state = np.concatenate([
                start, np.zeros(3), np.array([0., 0., 0., 1.])
            ])
        dummy_snap = StateSnapshot(time=0.0, state=snap_state, control=np.zeros(4))

        # ---- Perturbation candidates ----------------------------------------
        # Start with no perturbation, then try a grid of lateral+vertical offsets.
        steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        perturb_candidates: List[Tuple[float, float]] = [(0.0, 0.0)]
        for s in steps[1:]:
            for lat in ( s, -s):
                for vert in (0.0, s, -s):
                    perturb_candidates.append((lat, vert))

        max_attempts = min(len(perturb_candidates), 30)

        t0 = _time.time()
        best_positions: np.ndarray | None = None
        accepted_lat = accepted_vert = 0.0

        for attempt, (lat_off, vert_off) in enumerate(perturb_candidates[:max_attempts]):
            perturb = lat_off * lateral + vert_off * vertical
            before_gate = gate - offset * approach + perturb
            after_gate  = gate + offset * approach + perturb

            try:
                waypoints = [start, before_gate, gate, after_gate, goal]
                pos = self._min_snap_positions(waypoints, total_time=5.0, hz=10)
            except Exception as e:
                continue

            if self._trajectory_collision_free(pos):
                best_positions = pos
                accepted_lat   = lat_off
                accepted_vert  = vert_off
                break

        t1 = _time.time()
        feasible = best_positions is not None

        validation_ok, validation_meta = self._validate_trajectory_positions(
            best_positions, goal
        )
        feasible = bool(feasible and validation_ok)
        validation_meta["attempts"]        = attempt + 1 if best_positions is not None else max_attempts
        validation_meta["accepted_lat_m"]  = float(accepted_lat)
        validation_meta["accepted_vert_m"] = float(accepted_vert)
        validation_meta["before_gate_ned"] = (gate - offset * approach + accepted_lat * lateral + accepted_vert * vertical).tolist()
        validation_meta["after_gate_ned"]  = (gate + offset * approach + accepted_lat * lateral + accepted_vert * vertical).tolist()

        return RecoveryResult(
            feasible=feasible,
            safe_state=dummy_snap,
            goal_position=goal,
            trajectory_positions=best_positions,
            planning_time_s=t1 - t0,
            metadata=validation_meta,
        )

    def plan_recovery(
        self,
        failure_record: FailureRecord,
        goal_position_figs: np.ndarray,
    ) -> RecoveryResult:
        """
        Plan a safe trajectory from the last safe state to the goal.

        Parameters
        ----------
        failure_record : FailureRecord
            Output of the failure detector.
        goal_position_figs : array (3,)
            Goal position in FiGS (NED) coordinates.

        Returns
        -------
        RecoveryResult
        """
        import time as _time

        self._ensure_planner()

        safe_snap = failure_record.last_safe_state
        start_figs = safe_snap.state[:3]

        start_nerf = self._figs_pos_to_nerf(start_figs)
        goal_nerf = self._figs_pos_to_nerf(goal_position_figs)

        x0 = torch.tensor(start_nerf, dtype=torch.float32, device=self.device)
        xf = torch.tensor(goal_nerf, dtype=torch.float32, device=self.device)

        t0 = _time.time()
        try:
            result = self._planner.generate_path(x0, xf)
        except Exception as e:
            return RecoveryResult(
                feasible=False,
                safe_state=safe_snap,
                goal_position=goal_position_figs,
                planning_time_s=_time.time() - t0,
                metadata={"error": str(e)},
            )
        t1 = _time.time()

        feasible = result.get("feasible", False)

        traj_nerf = result.get("traj")
        traj_figs = None
        if traj_nerf is not None:
            traj_arr = np.asarray(traj_nerf)
            if traj_arr.ndim == 2 and traj_arr.shape[1] >= 3:
                traj_figs = np.array([self._nerf_pos_to_figs(p[:3]) for p in traj_arr])

        astar_path = result.get("path")
        if astar_path is not None:
            astar_path = np.asarray(astar_path)

        validation_ok, validation_meta = self._validate_recovery_trajectory(
            failure_record, traj_figs, goal_position_figs
        )
        feasible = bool(feasible and validation_ok)

        return RecoveryResult(
            feasible=feasible,
            safe_state=safe_snap,
            goal_position=goal_position_figs,
            trajectory_positions=traj_figs,
            astar_path=astar_path,
            polytopes=result.get("polytopes"),
            planning_time_s=t1 - t0,
            metadata={
                "times_astar": result.get("times_astar", 0),
                "times_collision_set": result.get("times_collision_set", 0),
                "times_polytope": result.get("times_polytope", 0),
                "times_opt": result.get("times_opt", 0),
                **validation_meta,
            },
        )

    def plan_recovery_from_state(
        self,
        start_position_figs: np.ndarray,
        goal_position_figs: np.ndarray,
    ) -> RecoveryResult:
        """Plan recovery directly from a position (no FailureRecord needed)."""
        import time as _time

        self._ensure_planner()

        start_nerf = self._figs_pos_to_nerf(start_position_figs)
        goal_nerf = self._figs_pos_to_nerf(goal_position_figs)

        x0 = torch.tensor(start_nerf, dtype=torch.float32, device=self.device)
        xf = torch.tensor(goal_nerf, dtype=torch.float32, device=self.device)

        dummy_snap = StateSnapshot(
            time=0.0,
            state=np.concatenate([start_position_figs, np.zeros(7)]),
            control=np.zeros(4),
        )

        t0 = _time.time()
        try:
            result = self._planner.generate_path(x0, xf)
        except Exception as e:
            return RecoveryResult(
                feasible=False,
                safe_state=dummy_snap,
                goal_position=goal_position_figs,
                planning_time_s=_time.time() - t0,
                metadata={"error": str(e)},
            )
        t1 = _time.time()

        feasible = result.get("feasible", False)
        traj_nerf = result.get("traj")
        traj_figs = None
        if traj_nerf is not None:
            traj_arr = np.asarray(traj_nerf)
            if traj_arr.ndim == 2 and traj_arr.shape[1] >= 3:
                traj_figs = np.array([self._nerf_pos_to_figs(p[:3]) for p in traj_arr])

        validation_ok, validation_meta = self._validate_trajectory_positions(
            traj_figs, goal_position_figs
        )
        feasible = bool(feasible and validation_ok)

        return RecoveryResult(
            feasible=feasible,
            safe_state=dummy_snap,
            goal_position=goal_position_figs,
            trajectory_positions=traj_figs,
            astar_path=np.asarray(result.get("path")) if result.get("path") is not None else None,
            polytopes=result.get("polytopes"),
            planning_time_s=t1 - t0,
            metadata={
                "times_astar": result.get("times_astar", 0),
                "times_collision_set": result.get("times_collision_set", 0),
                "times_polytope": result.get("times_polytope", 0),
                "times_opt": result.get("times_opt", 0),
                **validation_meta,
            },
        )

    # ------------------------------------------------------------------
    # Trajectory conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def trajectory_to_figs_waypoints(
        recovery_result: RecoveryResult,
        total_time: float = 5.0,
    ) -> Dict:
        """Convert a recovery trajectory to FiGS-compatible keyframes.

        Returns a ``course_config``-style dict that can be passed directly
        to ``VehicleRateMPC`` for tracking.
        """
        if not recovery_result.feasible or recovery_result.trajectory_positions is None:
            return None

        traj = recovery_result.trajectory_positions
        N = len(traj)

        keyframes = {}
        for i, pos in enumerate(traj):
            t = (i / max(N - 1, 1)) * total_time
            name = f"wp_{i:03d}"
            if i == 0 or i == N - 1:
                keyframes[name] = {
                    "t": t,
                    "fo": [
                        [float(pos[0]), 0.0],
                        [float(pos[1]), 0.0],
                        [float(pos[2]), 0.0],
                        [0.0, 0.0],
                    ],
                }
            else:
                keyframes[name] = {
                    "t": t,
                    "fo": [
                        [float(pos[0]), None, None, None],
                        [float(pos[1]), None, None, None],
                        [float(pos[2]), None, None, None],
                        [0.0, None, None, None],
                    ],
                }

        return {
            "waypoints": {"Nco": 6, "keyframes": keyframes},
            "forces": None,
        }

    @staticmethod
    def downsample_trajectory(
        positions: np.ndarray, max_waypoints: int = 20
    ) -> np.ndarray:
        """Reduce a dense trajectory to at most ``max_waypoints`` via uniform
        sub-sampling, always keeping the first and last point."""
        N = len(positions)
        if N <= max_waypoints:
            return positions

        indices = np.round(np.linspace(0, N - 1, max_waypoints)).astype(int)
        return positions[indices]
