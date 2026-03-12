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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

from vla_falsification.falsification.failure_detector import FailureRecord, StateSnapshot


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
    voxel_resolution: int = 100
    gate_position: Optional[List[float]] = None
    gate_pass_radius_m: float = 0.50
    goal_tolerance_m: float = 0.40
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

    def _trajectory_collision_free(
        self,
        positions_figs: np.ndarray,
        stride: int = 3,
        gate_pos_figs: np.ndarray | None = None,
        gate_clearance_m: float = 0.3,
    ) -> bool:
        """Return True iff no sampled position falls in the SplatNav voxel's
        non-navigable (obstacle-inflated) cells.

        Points within ``gate_clearance_m`` of ``gate_pos_figs`` are exempt from
        collision checking because the voxel grid cannot represent the gate
        opening (Gaussian bounding boxes span across it).

        Points are converted from FiGS NED to Nerfstudio before querying.
        """
        voxel = self._planner.gsplat_voxel
        for pos in positions_figs[::stride]:
            # Skip collision check near the gate opening
            if gate_pos_figs is not None:
                dist_to_gate = np.linalg.norm(pos - gate_pos_figs)
                if dist_to_gate < gate_clearance_m:
                    continue

            p_nerf = torch.tensor(
                self._figs_pos_to_nerf(pos), dtype=torch.float32, device=self.device
            )
            idx = voxel.get_indices(p_nerf)
            if voxel.non_navigable_grid[idx[0], idx[1], idx[2]]:
                gate_dist = (
                    float(np.linalg.norm(pos - gate_pos_figs))
                    if gate_pos_figs is not None else -1
                )
                logger.debug(
                    "Collision at FiGS=%s NS=%s idx=%s (%.3fm from gate)",
                    pos.tolist(),
                    self._figs_pos_to_nerf(pos).tolist(),
                    idx.cpu().numpy().tolist(),
                    gate_dist,
                )
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
        """Plan a safe path: start → gate → goal using SplatNav's polytope planner.

        Strategy
        --------
        1. Convert start, gate, goal to Nerfstudio frame.
        2. Plan two legs via ``SplatPlan.generate_path`` (A* → polytope
           decomposition → B-spline optimization):
           - Leg 1: start → gate
           - Leg 2: gate → goal
        3. Concatenate the two trajectories and convert back to FiGS NED.

        The A* path seed may need to project the gate position to the nearest
        navigable voxel (the gate opening is typically marked as occupied due
        to Gaussian bounding boxes spanning across it).  SplatNav handles this
        projection automatically.

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

        start = np.asarray(start_position_figs, dtype=float)
        gate  = np.asarray(gate_position_figs,  dtype=float)
        goal  = np.asarray(goal_position_figs,  dtype=float)

        # Convert to NS frame
        start_nerf = self._figs_pos_to_nerf(start)
        gate_nerf  = self._figs_pos_to_nerf(gate)
        goal_nerf  = self._figs_pos_to_nerf(goal)

        # Coordinate sanity check
        gate_rt = self._nerf_pos_to_figs(gate_nerf)
        rt_error = np.linalg.norm(gate - gate_rt)
        if rt_error > 1e-4:
            logger.warning(
                "Coordinate round-trip error %.6f for gate %s", rt_error, gate.tolist()
            )
        lb = np.array(self.config.env_lower_bound)
        ub = np.array(self.config.env_upper_bound)
        if np.any(gate_nerf < lb) or np.any(gate_nerf > ub):
            logger.warning(
                "Gate in NS frame %s outside voxel bounds [%s, %s]",
                gate_nerf.tolist(), lb.tolist(), ub.tolist(),
            )

        # Diagnostics
        voxel = self._planner.gsplat_voxel
        gate_nerf_t = torch.tensor(gate_nerf, dtype=torch.float32, device=self.device)
        gate_idx = voxel.get_indices(gate_nerf_t)
        gate_occupied = bool(voxel.non_navigable_grid[gate_idx[0], gate_idx[1], gate_idx[2]])
        if gate_occupied:
            logger.warning(
                "Gate voxel at NS=%s (idx=%s) is non-navigable — "
                "A* will project to nearest free voxel",
                gate_nerf.tolist(), gate_idx.cpu().numpy().tolist(),
            )

        # ---- Build dummy start snapshot for RecoveryResult ------------------
        if start_state_figs is not None and len(start_state_figs) == 10:
            snap_state = np.array(start_state_figs, dtype=float)
        else:
            snap_state = np.concatenate([
                start, np.zeros(3), np.array([0., 0., 0., 1.])
            ])
        dummy_snap = StateSnapshot(time=0.0, state=snap_state, control=np.zeros(4))

        # ---- Plan two legs via SplatNav -------------------------------------
        x_start = torch.tensor(start_nerf, dtype=torch.float32, device=self.device)
        x_gate  = torch.tensor(gate_nerf,  dtype=torch.float32, device=self.device)
        x_goal  = torch.tensor(goal_nerf,  dtype=torch.float32, device=self.device)

        t0 = _time.time()
        metadata: Dict = {}
        traj_figs = None

        try:
            logger.info("Planning leg 1: start → gate")
            leg1 = self._planner.generate_path(x_start, x_gate)
            logger.info("Planning leg 2: gate → goal")
            leg2 = self._planner.generate_path(x_gate, x_goal)
        except Exception as e:
            t1 = _time.time()
            logger.warning(
                "SplatNav planning failed: %s: %s. "
                "start_NS=%s, gate_NS=%s, goal_NS=%s",
                type(e).__name__, e,
                start_nerf.tolist(), gate_nerf.tolist(), goal_nerf.tolist(),
            )
            return RecoveryResult(
                feasible=False,
                safe_state=dummy_snap,
                goal_position=goal,
                planning_time_s=_time.time() - t0,
                metadata={"error": f"{type(e).__name__}: {e}"},
            )

        t1 = _time.time()

        # Combine leg trajectories (both in NS frame)
        leg1_feasible = leg1.get("feasible", False)
        leg2_feasible = leg2.get("feasible", False)
        feasible = leg1_feasible and leg2_feasible

        traj1_nerf = np.asarray(leg1.get("traj", []))
        traj2_nerf = np.asarray(leg2.get("traj", []))

        if traj1_nerf.ndim == 2 and traj2_nerf.ndim == 2:
            # Concatenate, dropping duplicate gate point
            combined_nerf = np.vstack([traj1_nerf, traj2_nerf[1:]])
            traj_figs = np.array([
                self._nerf_pos_to_figs(p[:3]) for p in combined_nerf
            ])
        elif traj1_nerf.ndim == 2:
            traj_figs = np.array([self._nerf_pos_to_figs(p[:3]) for p in traj1_nerf])
        elif traj2_nerf.ndim == 2:
            traj_figs = np.array([self._nerf_pos_to_figs(p[:3]) for p in traj2_nerf])

        # Collect A* paths for visualization
        astar_path = None
        p1 = leg1.get("path")
        p2 = leg2.get("path")
        if p1 is not None and p2 is not None:
            astar_path = np.vstack([np.asarray(p1), np.asarray(p2)[1:]])
        elif p1 is not None:
            astar_path = np.asarray(p1)
        elif p2 is not None:
            astar_path = np.asarray(p2)

        # Validate using the perturbed gate position
        saved_gate_position = self.config.gate_position
        self.config.gate_position = gate.tolist()
        validation_ok, validation_meta = self._validate_trajectory_positions(
            traj_figs, goal
        )
        self.config.gate_position = saved_gate_position
        feasible = bool(feasible and validation_ok)

        metadata.update(validation_meta)
        metadata["leg1_feasible"] = leg1_feasible
        metadata["leg2_feasible"] = leg2_feasible
        metadata["leg1_time_astar"] = leg1.get("times_astar", 0)
        metadata["leg2_time_astar"] = leg2.get("times_astar", 0)
        metadata["leg1_polytopes"] = leg1.get("num_polytopes", 0)
        metadata["leg2_polytopes"] = leg2.get("num_polytopes", 0)

        logger.info(
            "plan_via_gate: feasible=%s (leg1=%s, leg2=%s) in %.2fs. "
            "%d+%d polytopes, %d trajectory points",
            feasible, leg1_feasible, leg2_feasible, t1 - t0,
            leg1.get("num_polytopes", 0), leg2.get("num_polytopes", 0),
            len(traj_figs) if traj_figs is not None else 0,
        )

        return RecoveryResult(
            feasible=feasible,
            safe_state=dummy_snap,
            goal_position=goal,
            trajectory_positions=traj_figs,
            astar_path=astar_path,
            planning_time_s=t1 - t0,
            metadata=metadata,
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
