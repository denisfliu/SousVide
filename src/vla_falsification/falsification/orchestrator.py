"""
Falsification orchestrator.

Ties together:
- VLA policy rollout in FiGS (dual-camera sim loop)
- Perturbation engine (action / observation / environment)
- Failure detection (safety criteria → last safe state)
- SplatNav recovery (safe replanning from last safe state → goal)

The main entry point is ``FalsificationOrchestrator.run()``, which executes a
single episode and returns a ``FalsificationEpisode`` containing the full
trajectory, any failure record, and (if applicable) the recovery trajectory.

For batch runs, call ``run_campaign()`` which sweeps over seeds / perturbation
configs and aggregates results.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import figs.utilities.transform_helper as th
import figs.utilities.orientation_helper as oh
import figs.dynamics.quadcopter_specifications as qs
from figs.dynamics.external_forces import ExternalForces
from figs.simulator import Simulator

from vla_falsification.control.vla_policy import VLAPolicy, VLAPolicyConfig

logger = logging.getLogger(__name__)
from vla_falsification.falsification.failure_detector import (
    FailureDetector,
    FailureRecord,
    FailureType,
    SafetyCriterion,
    StateSnapshot,
    BoundsCriterion,
    VelocityCriterion,
    TiltCriterion,
    ProximityCollisionCriterion,
)
from vla_falsification.falsification.perturbations import PerturbationSuite
from vla_falsification.falsification.splatnav_recovery import (
    RecoveryConfig,
    RecoveryResult,
    SplatNavRecovery,
)


# ===================================================================
# Episode result data
# ===================================================================

@dataclass
class FalsificationEpisode:
    """Complete record of one falsification episode."""
    episode_id: int
    seed: int
    success: bool                                   # True = no failure detected
    trajectory: List[StateSnapshot] = field(default_factory=list)
    failure_record: Optional[FailureRecord] = None
    recovery_result: Optional[RecoveryResult] = None
    recovery_figs_data: Optional[Dict] = None       # FiGS rollout of recovery traj
    perturbation_config: Optional[Dict] = None
    wall_time_s: float = 0.0
    metadata: Dict = field(default_factory=dict)


# ===================================================================
# Orchestrator configuration
# ===================================================================

@dataclass
class OrchestratorConfig:
    """Top-level config for the falsification pipeline."""
    # Simulation
    t0: float = 0.0
    tf: float = 12.0
    frame_name: str = "carl"

    # Goal (NED coordinates for the FiGS sim)
    goal_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gate_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gate_pass_radius_m: float = 0.25

    # Initial state (NED)
    x0: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0, 1.0])

    # Dual-camera transforms (body → camera)
    Tc2b_forward: Optional[np.ndarray] = None
    Tc2b_downward: Optional[np.ndarray] = None

    # Safety bounds (NED)
    bounds_lower: List[float] = field(default_factory=lambda: [-5, -5, -5])
    bounds_upper: List[float] = field(default_factory=lambda: [5, 5, 5])
    max_speed: float = 5.0
    max_tilt_deg: float = 60.0
    robot_radius: float = 0.225
    gate_opening_half_w: float = 0.25
    gate_opening_half_h: float = 0.25
    gate_post_radius: float = 0.04

    # Failure detector
    safe_horizon: int = 3

    # Recovery
    enable_recovery: bool = True
    recovery_rollout: bool = True   # also roll out the recovery traj in FiGS
    recovery_total_time: float = 5.0

    # Coordinate transforms
    permutation: int = 5
    coordinate_transform_figs_to_nerf: Optional[np.ndarray] = None


# ===================================================================
# Orchestrator
# ===================================================================

class FalsificationOrchestrator:
    """
    Runs a VLA policy in FiGS with perturbations, detects failures,
    and plans recovery trajectories with SplatNav.

    Parameters
    ----------
    simulator : Simulator
        Pre-initialised FiGS simulator (with GSplat loaded).
    vla_policy : VLAPolicy
        The VLA model wrapped as a FiGS controller.
    perturbation_suite : PerturbationSuite
        Perturbations to apply during rollout.
    config : OrchestratorConfig
    splatnav_recovery : SplatNavRecovery, optional
        If provided, recovery planning is attempted on failure.
    """

    def __init__(
        self,
        simulator: Simulator,
        vla_policy: VLAPolicy,
        perturbation_suite: PerturbationSuite,
        config: OrchestratorConfig | None = None,
        splatnav_recovery: SplatNavRecovery | None = None,
    ):
        self.simulator = simulator
        self.vla_policy = vla_policy
        self.perturbations = perturbation_suite
        self.config = config or OrchestratorConfig()
        self.recovery = splatnav_recovery

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, episode_id: int = 0, seed: int = 0) -> FalsificationEpisode:
        """Execute one falsification episode.

        1. Reset perturbations with ``seed``
        2. (Optionally) apply environment perturbations to the Gaussian splat
        3. Run the VLA in the dual-camera sim loop, applying action + obs
           perturbations at each step
        4. If failure detected → use SplatNav to plan recovery
        5. (Optionally) roll out the recovery trajectory in FiGS for data
        """
        t_wall_start = time.time()
        cfg = self.config

        # --- Reset perturbations ------------------------------------------------
        self.perturbations.reset_all(seed)

        # --- Apply environment perturbations to the Gaussian splat -------------
        model_params_backup = self._apply_env_perturbations()

        # --- Build failure detector --------------------------------------------
        criteria: List[SafetyCriterion] = [
            BoundsCriterion(
                lower=np.array(cfg.bounds_lower),
                upper=np.array(cfg.bounds_upper),
            ),
            VelocityCriterion(max_speed=cfg.max_speed),
            TiltCriterion(max_tilt_deg=cfg.max_tilt_deg),
        ]
        gate_collision_criterion = self._build_gate_collision_criterion()
        if gate_collision_criterion is not None:
            criteria.append(gate_collision_criterion)
        detector = FailureDetector(criteria=criteria, safe_horizon=cfg.safe_horizon)

        # --- Run VLA in FiGS --------------------------------------------------
        trajectory, failure_record_inner = self._run_vla_loop(detector)

        # --- Analyse result ----------------------------------------------------
        failure_record = failure_record_inner
        recovery_result = None
        recovery_figs = None

        if detector.is_failed and failure_record is None:
            # Fallback: build a record from detector state
            last_snap = trajectory[-1] if trajectory else StateSnapshot(
                time=0, state=np.zeros(10), control=np.zeros(4))
            safe_snap = detector.last_safe_state or (trajectory[0] if trajectory else last_snap)
            failure_record = FailureRecord(
                failure_type=FailureType.CUSTOM,
                description="detected during rollout",
                failure_step=last_snap.step_index,
                failure_state=last_snap,
                last_safe_step=safe_snap.step_index,
                last_safe_state=safe_snap,
                trajectory_up_to_failure=list(trajectory),
            )

        goal_ned = np.array(cfg.goal_position)
        start_snap = trajectory[0] if trajectory else None
        start_ned = np.array(start_snap.state[:3]) if start_snap else goal_ned
        start_state_ned = np.array(start_snap.state) if start_snap else None
        perturbed_gate_ned = self._get_perturbed_gate_position_figs()

        if failure_record is not None:
            if cfg.enable_recovery and self.recovery is not None:
                recovery_result = self.recovery.plan_via_gate(
                    start_position_figs=start_ned,
                    gate_position_figs=perturbed_gate_ned,
                    goal_position_figs=goal_ned,
                    start_state_figs=start_state_ned,
                )

                if cfg.recovery_rollout and recovery_result.feasible:
                    recovery_figs = self._rollout_recovery(recovery_result)
        else:
            # Explicitly mark "did not pass through gate" as a failure mode.
            if trajectory and not self._trajectory_passes_gate(trajectory):
                fail_snap = trajectory[-1]
                safe_snap = detector.last_safe_state or trajectory[0]
                failure_record = FailureRecord(
                    failure_type=FailureType.MISSED_GATE,
                    description=(
                        "Trajectory did not pass within "
                        f"{cfg.gate_pass_radius_m:.3f}m of gate waypoint"
                    ),
                    failure_step=fail_snap.step_index,
                    failure_state=fail_snap,
                    last_safe_step=safe_snap.step_index,
                    last_safe_state=safe_snap,
                    trajectory_up_to_failure=list(trajectory),
                    metadata={
                        "gate_position_ned": np.array(cfg.gate_position).tolist(),
                        "gate_pass_radius_m": cfg.gate_pass_radius_m,
                    },
                )
                if cfg.enable_recovery and self.recovery is not None:
                    recovery_result = self.recovery.plan_via_gate(
                        start_position_figs=start_ned,
                        gate_position_figs=perturbed_gate_ned,
                        goal_position_figs=goal_ned,
                        start_state_figs=start_state_ned,
                    )
                    if cfg.recovery_rollout and recovery_result.feasible:
                        recovery_figs = self._rollout_recovery(recovery_result)

        # --- Restore environment -----------------------------------------------
        self._restore_env(model_params_backup)

        return FalsificationEpisode(
            episode_id=episode_id,
            seed=seed,
            success=failure_record is None,
            trajectory=trajectory,
            failure_record=failure_record,
            recovery_result=recovery_result,
            recovery_figs_data=recovery_figs,
            perturbation_config=None,
            wall_time_s=time.time() - t_wall_start,
        )

    # ------------------------------------------------------------------
    # VLA simulation loop (dual camera, with perturbations)
    # ------------------------------------------------------------------

    def _run_vla_loop(
        self, detector: FailureDetector
    ) -> Tuple[List[StateSnapshot], Optional[FailureRecord]]:
        """Run the VLA in FiGS with dual cameras, perturbations, and live
        failure detection.  Returns the trajectory and (if failed) the record."""
        cfg = self.config
        sim = self.simulator

        conFiG = sim.conFiG
        Rout = conFiG["rollout"]
        Spec = qs.generate_specifications(conFiG["frame"])

        # Pass frame spec to VLA policy so it can build its MPC
        self.vla_policy._frame_spec = Spec

        fex = ExternalForces(conFiG["forces"])

        nx = Spec["nx"]
        m, kt = Spec["m"], Spec["kt"]
        g, Nrtr = Spec["g"], Spec["Nrtr"]

        Tc2b_fwd = cfg.Tc2b_forward if cfg.Tc2b_forward is not None else Spec["Tc2b"]
        Tc2b_dwn = cfg.Tc2b_downward if cfg.Tc2b_downward is not None else Spec["Tc2b"]

        camera_fwd = sim.gsplat.generate_output_camera(Spec["camera"])
        camera_dwn = sim.gsplat.generate_output_camera(Spec["camera"])

        hz_sim = Rout["frequency"]
        n_sim2ctl = int(hz_sim / self.vla_policy.hz)
        dt = np.round(cfg.tf - cfg.t0, 5)
        Nsim = int(dt * hz_sim)
        Nctl = int(dt * self.vla_policy.hz)

        mu_md, std_md = np.zeros(nx), np.zeros(nx)
        mu_sn, std_sn = np.zeros(nx), np.zeros(nx)

        xcr = np.array(cfg.x0, dtype=float)
        xpr = xcr.copy()
        ucr = np.array([-(m * g) / (Nrtr * kt), 0.0, 0.0, 0.0])

        trajectory: List[StateSnapshot] = []
        failure_record: Optional[FailureRecord] = None
        tau_cr = np.zeros(3)

        logger.info(
            "VLA loop: %d sim steps, %d control steps, hz_sim=%s, n_sim2ctl=%d",
            Nsim, Nctl, hz_sim, n_sim2ctl,
        )

        for i in range(Nsim):
            tcr = cfg.t0 + i / hz_sim
            fcr = fex.get_forces(xcr[0:6], noisy=True)
            pcr = np.hstack((m, kt, fcr))
            fts = np.hstack((fcr, tau_cr))

            if i % n_sim2ctl == 0:
                k = i // n_sim2ctl

                # --- Observation perturbation on camera poses ---
                Tc2b_fwd_pert = Tc2b_fwd
                Tc2b_dwn_pert = Tc2b_dwn
                if len(self.perturbations.observation_camera) > 0:
                    Tc2b_fwd_pert = self.perturbations.observation_camera.apply(Tc2b_fwd.copy())
                    Tc2b_dwn_pert = self.perturbations.observation_camera.apply(Tc2b_dwn.copy())

                Tb2w = th.x_to_T(xcr)
                Tc2w_fwd = Tb2w @ Tc2b_fwd_pert
                Tc2w_dwn = Tb2w @ Tc2b_dwn_pert

                t_render_start = time.time()
                rgb_fwd, dpt_fwd = sim.gsplat.render_rgb(camera_fwd, Tc2w_fwd)
                rgb_dwn, dpt_dwn = sim.gsplat.render_rgb(camera_dwn, Tc2w_dwn)
                t_render_end = time.time()

                # --- Observation perturbation on images ---
                if len(self.perturbations.observation_image) > 0:
                    rgb_fwd = self.perturbations.observation_image.apply(rgb_fwd)
                    rgb_dwn = self.perturbations.observation_image.apply(rgb_dwn)

                # --- Observation perturbation on state ---
                xsn = xcr + np.random.normal(loc=mu_sn, scale=std_sn)
                xsn[6:10] = oh.obedient_quaternion(xsn[6:10], xpr[6:10])
                if len(self.perturbations.observation_state) > 0:
                    xsn = self.perturbations.observation_state.apply(xsn)

                # Inject downward image before calling control
                self.vla_policy.set_downward_image(rgb_dwn)
                t_ctrl_start = time.time()
                ucr_raw, tsol = self.vla_policy.control(tcr, xsn, ucr, rgb_fwd, dpt_fwd, fts)
                t_ctrl_end = time.time()

                logger.debug(
                    "step %d/%d  t=%.2f  pos=%s  speed=%.3f  "
                    "render=%.2fs  ctrl=%.2fs  queue=%s",
                    k, Nctl, tcr, xcr[:3],
                    np.linalg.norm(xcr[3:6]),
                    t_render_end - t_render_start,
                    t_ctrl_end - t_ctrl_start,
                    tsol.get("queue_len", "?"),
                )

                # --- Action perturbation ---
                ucr = ucr_raw
                if len(self.perturbations.action) > 0:
                    ucr = self.perturbations.action.apply(ucr)

                # --- Record snapshot & check safety ---
                snap = StateSnapshot(
                    time=tcr,
                    state=xcr.copy(),
                    control=ucr.copy(),
                    rgb_forward=rgb_fwd,
                    rgb_downward=rgb_dwn,
                    depth=dpt_fwd,
                    step_index=k,
                )
                trajectory.append(snap)

                failed, record = detector.step(snap)
                if failed:
                    failure_record = record
                    break

            xpr = xcr
            xcr = sim.solver.simulate(x=xcr, u=ucr, p=pcr)
            xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
            xcr[6:10] = oh.obedient_quaternion(xcr[6:10], xpr[6:10])

        return trajectory, failure_record

    # ------------------------------------------------------------------
    # Environment perturbation helpers
    # ------------------------------------------------------------------

    def _apply_env_perturbations(self) -> Dict[str, Any]:
        """Apply Gaussian splat perturbations and return originals for restore."""
        backup: Dict[str, Any] = {}
        model = self.simulator.gsplat.pipeline.model

        if len(self.perturbations.environment_means) > 0:
            means_data = model.means.data
            quats_data = model.quats.data
            backup["means"] = means_data.clone()

            means_out = means_data
            quats_out = quats_data
            quats_changed = False

            for perturb in self.perturbations.environment_means.perturbations:
                means_out = perturb.apply(means_out)
                if hasattr(perturb, "apply_quats"):
                    if "quats" not in backup:
                        backup["quats"] = quats_data.clone()
                    quats_out = perturb.apply_quats(quats_out)
                    quats_changed = True

            model.means.data = means_out
            if quats_changed:
                model.quats.data = quats_out

        if len(self.perturbations.environment_scales) > 0:
            scales = model.scales
            backup["scales"] = scales.data.clone()
            model.scales.data = self.perturbations.environment_scales.apply(scales.data)

        if len(self.perturbations.environment_opacities) > 0:
            opacities = model.opacities
            backup["opacities"] = opacities.data.clone()
            model.opacities.data = self.perturbations.environment_opacities.apply(opacities.data)

        return backup

    def _build_gate_collision_criterion(self) -> ProximityCollisionCriterion | None:
        """Build a gate-frame collision check that follows the sampled gate perturbation."""
        cfg = self.config
        if cfg.coordinate_transform_figs_to_nerf is None:
            return None

        # Base gate-frame spheres in FiGS/NED coordinates.
        cx, cy, cz = np.asarray(cfg.gate_position, dtype=float)
        hw = float(cfg.gate_opening_half_w)
        hh = float(cfg.gate_opening_half_h)
        base_centers_figs = np.array(
            [
                [cx, cy - hw, cz],
                [cx, cy + hw, cz],
                [cx, cy, cz + hh],
                [cx, cy, cz - hh],
                [cx, cy - hw, cz + hh],
                [cx, cy + hw, cz + hh],
                [cx, cy - hw, cz - hh],
                [cx, cy + hw, cz - hh],
            ],
            dtype=float,
        )

        t_figs_to_nerf = np.asarray(cfg.coordinate_transform_figs_to_nerf, dtype=float)
        t_nerf_to_figs = np.linalg.inv(t_figs_to_nerf)
        base_centers_nerf = np.array(
            [(t_figs_to_nerf @ np.append(p, 1.0))[:3] for p in base_centers_figs],
            dtype=float,
        )
        gate_center_nerf = (t_figs_to_nerf @ np.append(np.asarray(cfg.gate_position, dtype=float), 1.0))[:3]

        transformed_centers_nerf = base_centers_nerf
        for perturb in self.perturbations.environment_means.perturbations:
            if not hasattr(perturb, "sampled_transform"):
                continue
            translation, rotation = perturb.sampled_transform
            if translation is None or rotation is None:
                continue
            transformed_centers_nerf = (
                (transformed_centers_nerf - gate_center_nerf[None, :]) @ rotation.T
                + gate_center_nerf[None, :]
                + np.asarray(translation, dtype=float)[None, :]
            )

        transformed_centers_figs = np.array(
            [(t_nerf_to_figs @ np.append(p, 1.0))[:3] for p in transformed_centers_nerf],
            dtype=float,
        )
        obstacles = [(center, float(cfg.gate_post_radius)) for center in transformed_centers_figs]
        return ProximityCollisionCriterion(
            obstacles=obstacles,
            robot_radius=float(cfg.robot_radius),
        )

    def _restore_env(self, backup: Dict[str, Any]) -> None:
        """Restore Gaussian splat parameters from backup."""
        if not backup:
            return
        model = self.simulator.gsplat.pipeline.model
        if "means" in backup:
            model.means.data = backup["means"]
        if "quats" in backup:
            model.quats.data = backup["quats"]
        if "scales" in backup:
            model.scales.data = backup["scales"]
        if "opacities" in backup:
            model.opacities.data = backup["opacities"]

    def _get_perturbed_gate_position_figs(self) -> np.ndarray:
        """Return the perturbed gate centre in FiGS (NED) coordinates.

        Applies the same rotation+translation that ``_build_gate_collision_criterion``
        applies to the individual gate obstacle spheres, but to the scalar gate
        centre point.  If no coordinate transform is available, returns the
        nominal gate position.
        """
        cfg = self.config
        nominal_figs = np.asarray(cfg.gate_position, dtype=float)
        if cfg.coordinate_transform_figs_to_nerf is None:
            return nominal_figs

        t_figs_to_nerf = np.asarray(cfg.coordinate_transform_figs_to_nerf, dtype=float)
        t_nerf_to_figs = np.linalg.inv(t_figs_to_nerf)

        gate_nerf = (t_figs_to_nerf @ np.append(nominal_figs, 1.0))[:3]
        perturbed_nerf = gate_nerf.copy()

        for perturb in self.perturbations.environment_means.perturbations:
            if not hasattr(perturb, "sampled_transform"):
                continue
            translation, rotation = perturb.sampled_transform
            if translation is None or rotation is None:
                continue
            # Rotate around the gate centre, then translate
            perturbed_nerf = (
                rotation @ (perturbed_nerf - gate_nerf)
                + gate_nerf
                + np.asarray(translation, dtype=float)
            )

        return (t_nerf_to_figs @ np.append(perturbed_nerf, 1.0))[:3]

    def _trajectory_passes_gate(self, trajectory: List[StateSnapshot]) -> bool:
        """Returns True if any state gets sufficiently close to the gate."""
        gate_pos = np.asarray(self.config.gate_position, dtype=float)
        radius = float(self.config.gate_pass_radius_m)
        for snap in trajectory:
            if np.linalg.norm(snap.state[:3] - gate_pos) <= radius:
                return True
        return False

    # ------------------------------------------------------------------
    # Recovery rollout
    # ------------------------------------------------------------------

    def _rollout_recovery(self, recovery_result: RecoveryResult) -> Dict:
        """Roll out the recovery trajectory in FiGS using MPC tracking,
        producing corrective training data."""
        from figs.control.vehicle_rate_mpc import VehicleRateMPC

        cfg = self.config

        course_config = SplatNavRecovery.trajectory_to_figs_waypoints(
            recovery_result, total_time=cfg.recovery_total_time,
        )
        if course_config is None:
            return {"error": "no feasible trajectory to roll out"}

        # Downsample if too many waypoints
        if recovery_result.trajectory_positions is not None:
            n_wp = len(recovery_result.trajectory_positions)
            if n_wp > 30:
                ds_traj = SplatNavRecovery.downsample_trajectory(
                    recovery_result.trajectory_positions, max_waypoints=20
                )
                course_config = SplatNavRecovery.trajectory_to_figs_waypoints(
                    RecoveryResult(
                        feasible=True,
                        safe_state=recovery_result.safe_state,
                        goal_position=recovery_result.goal_position,
                        trajectory_positions=ds_traj,
                    ),
                    total_time=cfg.recovery_total_time,
                )

        policy_cfg = {
            "plan": {"kT": 10.0, "use_l2_time": False},
            "track": {
                "hz": 10,
                "horizon": 40,
                "Qk": [5e-1]*10,
                "Rk": [1e0, 1e-1, 1e-1, 1e-2],
                "QN": [5e-2]*10,
                "Ws": [5e-2]*10,
                "bounds": {"lower": [-1.0, -5.0, -5.0, -5.0],
                           "upper": [0.0, 5.0, 5.0, 5.0]},
            },
        }

        # Build the recovery MPC in an isolated temp directory so its compiled
        # ACADOS shared library (.so) doesn't collide with the VLA policy's
        # already-loaded solver (Linux dlopen caches by path; both would
        # otherwise write to the same ./c_generated_code/ path).
        import tempfile
        _orig_cwd = os.getcwd()
        _tmp_dir = tempfile.mkdtemp(prefix="recovery_mpc_")
        try:
            os.chdir(_tmp_dir)
            controller = VehicleRateMPC(
                policy=policy_cfg,
                course=course_config,
                frame=cfg.frame_name,
            )
        finally:
            os.chdir(_orig_cwd)

        safe_state = recovery_result.safe_state.state
        t0 = 0.0
        tf = cfg.recovery_total_time

        Tro, Xro, Uro, Wro, Rgb, Dpt, Tsol = self.simulator.simulate(
            controller, t0, tf, safe_state
        )

        return {
            "Tro": Tro, "Xro": Xro, "Uro": Uro, "Wro": Wro,
            "Rgb": Rgb, "Dpt": Dpt, "Tsol": Tsol,
            "start_state": safe_state,
            "goal_position": recovery_result.goal_position,
        }


# ===================================================================
# Campaign runner (batch falsification)
# ===================================================================

def run_campaign(
    orchestrator: FalsificationOrchestrator,
    num_episodes: int = 100,
    seed_offset: int = 0,
    perturbation_configs: Sequence[Dict] | None = None,
    progress_callback: Callable[[int, FalsificationEpisode], None] | None = None,
) -> List[FalsificationEpisode]:
    """Run multiple falsification episodes and collect results.

    Parameters
    ----------
    orchestrator : FalsificationOrchestrator
    num_episodes : int
    seed_offset : int
        Seeds will be ``seed_offset + i`` for episode ``i``.
    perturbation_configs : list of dicts, optional
        If provided, each episode ``i`` uses config ``i % len(configs)``.
        Otherwise the orchestrator's existing perturbation suite is reused.
    progress_callback : callable(episode_id, episode), optional
        Called after each episode completes.

    Returns
    -------
    episodes : list of FalsificationEpisode
    """
    from vla_falsification.falsification.perturbations import build_perturbation_suite

    episodes: List[FalsificationEpisode] = []

    for i in range(num_episodes):
        seed = seed_offset + i

        if perturbation_configs:
            cfg = perturbation_configs[i % len(perturbation_configs)]
            orchestrator.perturbations = build_perturbation_suite(cfg)

        episode = orchestrator.run(episode_id=i, seed=seed)
        episodes.append(episode)

        if progress_callback is not None:
            progress_callback(i, episode)

    return episodes


def summarize_campaign(episodes: List[FalsificationEpisode]) -> Dict:
    """Compute aggregate statistics from a campaign."""
    n = len(episodes)
    n_fail = sum(1 for e in episodes if not e.success)
    n_success = n - n_fail
    n_recovered = sum(
        1 for e in episodes
        if e.recovery_result is not None and e.recovery_result.feasible
    )

    failure_types: Dict[str, int] = {}
    for e in episodes:
        if e.failure_record is not None:
            ft = e.failure_record.failure_type.name
            failure_types[ft] = failure_types.get(ft, 0) + 1

    avg_wall = np.mean([e.wall_time_s for e in episodes]) if episodes else 0
    fail_steps = [
        e.failure_record.failure_step
        for e in episodes if e.failure_record is not None
    ]

    return {
        "total_episodes": n,
        "successes": n_success,
        "failures": n_fail,
        "failure_rate": n_fail / n if n > 0 else 0,
        "recovered": n_recovered,
        "recovery_rate": n_recovered / max(n_fail, 1),
        "failure_types": failure_types,
        "avg_failure_step": float(np.mean(fail_steps)) if fail_steps else None,
        "avg_wall_time_s": float(avg_wall),
    }
