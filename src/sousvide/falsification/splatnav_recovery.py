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
    robot_radius: float = 0.05
    vmax: float = 2.0
    amax: float = 3.0
    spline_degree: int = 6
    n_spline_sections: int = 10
    env_lower_bound: List[float] = field(default_factory=lambda: [-3.0, -3.0, -3.0])
    env_upper_bound: List[float] = field(default_factory=lambda: [3.0, 3.0, 3.0])
    voxel_resolution: int = 100


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

    # ------------------------------------------------------------------
    # Recovery planning
    # ------------------------------------------------------------------

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
            if traj_arr.ndim == 2 and traj_arr.shape[1] == 3:
                traj_figs = np.array([self._nerf_pos_to_figs(p) for p in traj_arr])

        astar_path = result.get("path")
        if astar_path is not None:
            astar_path = np.asarray(astar_path)

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
            if traj_arr.ndim == 2 and traj_arr.shape[1] == 3:
                traj_figs = np.array([self._nerf_pos_to_figs(p) for p in traj_arr])

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
