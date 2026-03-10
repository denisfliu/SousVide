#!/usr/bin/env python3
"""Plan and save a recovery trajectory for a saved falsification episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CC", "gcc-11")
os.environ.setdefault("CXX", "g++-11")

# Ensure repo root is on sys.path for standalone execution
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "external" / "FiGS" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "external" / "splatnav"))

from vla_falsification.utilities.coordinate_transform import create_transformer_for_scene
from vla_falsification.falsification.config import apply_gate_preset, convert_from_ned_to_zup, convert_to_ned, load_config
from vla_falsification.falsification.failure_detector import FailureRecord, FailureType, StateSnapshot
from vla_falsification.falsification.orchestrator import FalsificationOrchestrator, OrchestratorConfig
from vla_falsification.falsification.perturbations import build_perturbation_suite
from vla_falsification.falsification.splatnav_recovery import RecoveryConfig, RecoveryResult, SplatNavRecovery

import figs.utilities.config_helper as ch
from figs.render.gsplat import GSplat
from figs.simulator import Simulator

import torch
from ellipsoids.covariance_utils import compute_cov
from splat.splat_utils import GSplatLoader
from splatplan.spline_utils import SplinePlanner
from splatplan.splatplan import SplatPlan


def _build_camera_transforms() -> tuple[np.ndarray, np.ndarray]:
    tc2b_forward_base = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, -0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    r_z_180 = np.eye(4)
    r_z_180[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    r_y_180 = np.eye(4)
    r_y_180[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    r_y_90 = np.eye(4)
    r_y_90[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]])
    tc2b_forward = tc2b_forward_base @ r_z_180 @ r_y_180 @ r_y_90

    tc2b_downward_base = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    r_x_180 = np.eye(4)
    r_x_180[:3, :3] = np.diag([1.0, -1.0, -1.0])
    r_z_90_d = np.eye(4)
    r_z_90_d[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    tc2b_downward = tc2b_downward_base @ r_x_180 @ r_z_90_d
    return tc2b_forward, tc2b_downward


def _build_figs_to_nerf_transform(
    scene_key: str,
    permutation: int,
    config_yml_path: "str | Path | None" = None,
) -> np.ndarray:
    """Compose FiGS NED -> MOCAP Z-up -> COLMAP -> Nerfstudio-internal transform.

    The nerfstudio dataparser applies an additional scale+rotation stored in
    ``dataparser_transforms.json`` next to ``config.yml``.  When
    ``config_yml_path`` is given we load and compose that extra step so the
    result maps all the way to the NS-internal frame used by the SplatNav voxel.
    """
    import json as _json

    transformer = create_transformer_for_scene(scene_key)

    # FiGS NED -> MOCAP Z-up
    t_figs_to_mocap = np.eye(4)
    if permutation == 5:
        t_figs_to_mocap[:3, :3] = np.diag([1.0, -1.0, -1.0])
    elif permutation == 0:
        t_figs_to_mocap[:3, :3] = np.diag([1.0, 1.0, -1.0])
    elif permutation == 2:
        t_figs_to_mocap[:3, :3] = np.diag([-1.0, -1.0, -1.0])
    else:
        t_figs_to_mocap[:3, :3] = np.diag([1.0, 1.0, -1.0])

    # MOCAP Z-up -> COLMAP  (Sim(3) inverse)
    t_mocap_to_colmap = np.eye(4)
    t_mocap_to_colmap[:3, :3] = transformer.s_inv * transformer.R_inv
    t_mocap_to_colmap[:3, 3] = transformer.t_inv

    T = t_mocap_to_colmap @ t_figs_to_mocap

    # COLMAP -> Nerfstudio-internal  (dataparser_transforms.json)
    if config_yml_path is not None:
        dp_path = Path(config_yml_path).parent / "dataparser_transforms.json"
        if dp_path.exists():
            dp = _json.loads(dp_path.read_text())
            dp_mat = np.array(dp["transform"])   # (3, 4)
            dp_scale = float(dp["scale"])
            t_colmap_to_ns = np.eye(4)
            t_colmap_to_ns[:3, :3] = dp_scale * dp_mat[:, :3]
            t_colmap_to_ns[:3, 3] = dp_scale * dp_mat[:, 3]
            T = t_colmap_to_ns @ T

    return T


def _apply_environment_perturbations_to_ns_model(model, suite) -> dict:
    result: dict = {}

    if len(suite.environment_means) > 0:
        means_out = model.means.data
        quats_out = model.quats.data
        for perturb in suite.environment_means.perturbations:
            means_out = perturb.apply(means_out)
            if hasattr(perturb, "apply_quats"):
                quats_out = perturb.apply_quats(quats_out)
            if getattr(perturb, "_translation", None) is not None:
                result["translation_xyz_m"] = np.asarray(perturb._translation).tolist()
            if getattr(perturb, "_yaw_rad", None) is not None:
                result["yaw_deg"] = float(np.degrees(perturb._yaw_rad))
        model.means.data = means_out
        model.quats.data = quats_out

    if len(suite.environment_scales) > 0:
        model.scales.data = suite.environment_scales.apply(model.scales.data)

    if len(suite.environment_opacities) > 0:
        model.opacities.data = suite.environment_opacities.apply(model.opacities.data)

    return result


def _apply_environment_perturbations_to_splatnav(gsplat_loader: GSplatLoader, suite) -> dict:
    result: dict = {}

    if len(suite.environment_means) > 0:
        means_out = gsplat_loader.means
        rots_out = gsplat_loader.rots
        for perturb in suite.environment_means.perturbations:
            means_out = perturb.apply(means_out)
            if hasattr(perturb, "apply_quats"):
                rots_out = perturb.apply_quats(rots_out)
            if getattr(perturb, "_translation", None) is not None:
                result["translation_xyz_m"] = np.asarray(perturb._translation).tolist()
            if getattr(perturb, "_yaw_rad", None) is not None:
                result["yaw_deg"] = float(np.degrees(perturb._yaw_rad))
        gsplat_loader.means = means_out
        gsplat_loader.rots = rots_out
        gsplat_loader.covs_inv = compute_cov(gsplat_loader.rots, 1.0 / gsplat_loader.scales)
        gsplat_loader.covs = compute_cov(gsplat_loader.rots, gsplat_loader.scales)

    return result


def _build_perturbed_recovery(cfg: dict, suite) -> tuple[SplatNavRecovery, dict]:
    config_yml_path = cfg["scene"].get("config_yml")
    t_figs_to_nerf = _build_figs_to_nerf_transform(
        cfg["scene"]["scene_key"],
        cfg["simulation"]["permutation"],
        config_yml_path=config_yml_path,
    )
    rec_cfg = RecoveryConfig(
        robot_radius=cfg["recovery"]["robot_radius"],
        vmax=cfg["recovery"]["vmax"],
        amax=cfg["recovery"]["amax"],
        env_lower_bound=cfg["recovery"].get("env_lower_bound", [-0.5, -0.5, -0.5]),
        env_upper_bound=cfg["recovery"].get("env_upper_bound", [0.5, 0.5, 0.5]),
        voxel_resolution=cfg["recovery"].get("voxel_resolution", 150),
        gate_position=convert_to_ned(
            cfg["simulation"]["gate_position_zup"],
            cfg["simulation"]["permutation"],
        ).tolist(),
        gate_pass_radius_m=cfg["simulation"]["gate_pass_radius_m"],
    )
    recovery = SplatNavRecovery(
        gsplat_path=cfg["scene"]["splatnav_gsplat_path"],
        config=rec_cfg,
        coordinate_transform=t_figs_to_nerf,
    )

    gsplat_loader = GSplatLoader(recovery.gsplat_path, recovery.device)
    perturb_info = _apply_environment_perturbations_to_splatnav(gsplat_loader, suite)

    robot_config = {
        "radius": rec_cfg.robot_radius,
        "vmax": rec_cfg.vmax,
        "amax": rec_cfg.amax,
    }
    env_config = {
        "lower_bound": torch.tensor(rec_cfg.env_lower_bound, device=recovery.device),
        "upper_bound": torch.tensor(rec_cfg.env_upper_bound, device=recovery.device),
        "resolution": rec_cfg.voxel_resolution,
    }
    spline_planner = SplinePlanner(
        spline_deg=rec_cfg.spline_degree,
        N_sec=rec_cfg.n_spline_sections,
        device=recovery.device,
    )
    recovery._gsplat = gsplat_loader
    recovery._planner = SplatPlan(gsplat_loader, robot_config, env_config, spline_planner, recovery.device)
    return recovery, perturb_info


def _load_episode_state(ep_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with open(ep_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    traj = np.load(ep_dir / "trajectory.npz")
    return metadata, traj["times"], traj["states"], traj["controls"]


def _build_failure_record(
    metadata: dict,
    times: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    override_safe_step: int | None = None,
) -> FailureRecord:
    failure = metadata["failure"]
    failure_step = int(failure["failure_step"])
    last_safe_step = int(failure["last_safe_step"] if override_safe_step is None else override_safe_step)

    failure_snap = StateSnapshot(
        time=float(times[failure_step]),
        state=states[failure_step].copy(),
        control=controls[min(failure_step, len(controls) - 1)].copy(),
        step_index=failure_step,
    )
    safe_snap = StateSnapshot(
        time=float(times[last_safe_step]),
        state=states[last_safe_step].copy(),
        control=controls[min(last_safe_step, len(controls) - 1)].copy(),
        step_index=last_safe_step,
    )

    trajectory_up_to_failure = [
        StateSnapshot(
            time=float(times[i]),
            state=states[i].copy(),
            control=controls[min(i, len(controls) - 1)].copy(),
            step_index=i,
        )
        for i in range(failure_step + 1)
    ]

    return FailureRecord(
        failure_type=FailureType[failure["type"]],
        description=failure["description"],
        failure_step=failure_step,
        failure_state=failure_snap,
        last_safe_step=last_safe_step,
        last_safe_state=safe_snap,
        trajectory_up_to_failure=trajectory_up_to_failure,
    )


def _compute_perturbed_gate_ned(cfg: dict, suite, t_figs_to_nerf: np.ndarray) -> np.ndarray:
    """Return the perturbed gate centre in FiGS (NED) coordinates.

    Applies the same rotation + translation that ``GateRigidTransform`` applies
    to the Gaussian means, but to the scalar gate centre.  Rotation is around
    the nominal gate centre (approximation for the Gaussian centroid).
    """
    gate_ned = convert_to_ned(
        cfg["simulation"]["gate_position_zup"],
        cfg["simulation"]["permutation"],
    )
    t_nerf_to_figs = np.linalg.inv(t_figs_to_nerf)
    gate_nerf_nom = (t_figs_to_nerf @ np.append(gate_ned, 1.0))[:3]
    gate_nerf = gate_nerf_nom.copy()

    for perturb in suite.environment_means.perturbations:
        translation = getattr(perturb, "_translation", None)
        rotation    = getattr(perturb, "_rotation_np", None)
        if translation is None or rotation is None:
            continue
        gate_nerf = (
            rotation @ (gate_nerf - gate_nerf_nom)
            + gate_nerf_nom
            + np.asarray(translation, dtype=float)
        )

    return (t_nerf_to_figs @ np.append(gate_nerf, 1.0))[:3]


def _rollout_recovery(cfg: dict, safe_result: RecoveryResult, suite) -> dict:
    config_yml = Path(cfg["scene"]["config_yml"])
    saved_cwd = Path.cwd()
    os.chdir(config_yml.parent)
    gsplat_obj = GSplat(config_yml)
    os.chdir(saved_cwd)
    _apply_environment_perturbations_to_ns_model(gsplat_obj.pipeline.model, suite)

    simulator = Simulator(
        gsplat=gsplat_obj,
        method="eval_single",
        frame=cfg["simulation"]["frame_name"],
    )
    tc2b_forward, tc2b_downward = _build_camera_transforms()
    goal_ned = convert_to_ned(cfg["simulation"]["goal_position_zup"], cfg["simulation"]["permutation"])
    gate_ned = convert_to_ned(cfg["simulation"]["gate_position_zup"], cfg["simulation"]["permutation"])
    x0 = safe_result.safe_state.state

    orch_cfg = OrchestratorConfig(
        t0=0.0,
        tf=cfg["simulation"]["tf"],
        frame_name=cfg["simulation"]["frame_name"],
        goal_position=goal_ned.tolist(),
        gate_position=gate_ned.tolist(),
        gate_pass_radius_m=cfg["simulation"]["gate_pass_radius_m"],
        x0=x0.tolist(),
        Tc2b_forward=tc2b_forward,
        Tc2b_downward=tc2b_downward,
        bounds_lower=cfg["safety"]["bounds_lower"],
        bounds_upper=cfg["safety"]["bounds_upper"],
        max_speed=cfg["safety"]["max_speed"],
        max_tilt_deg=cfg["safety"]["max_tilt_deg"],
        safe_horizon=cfg["safety"]["safe_horizon"],
        enable_recovery=True,
        recovery_rollout=True,
        recovery_total_time=cfg["recovery"]["recovery_total_time"],
        permutation=cfg["simulation"]["permutation"],
    )
    orchestrator = FalsificationOrchestrator(
        simulator=simulator,
        vla_policy=None,
        perturbation_suite=suite,
        config=orch_cfg,
        splatnav_recovery=None,
    )
    return orchestrator._rollout_recovery(safe_result)


def _save_outputs(
    results_dir: Path,
    episode_id: int,
    metadata: dict,
    cfg: dict,
    recovery_result: RecoveryResult,
    recovery_figs_data: dict | None,
    perturb_info: dict,
) -> None:
    ep_dir = results_dir / "episodes" / f"episode_{episode_id:05d}"
    perm = cfg["simulation"]["permutation"]

    recovery_ned = recovery_result.trajectory_positions
    recovery_pos_mocap = np.array([convert_from_ned_to_zup(p, perm) for p in recovery_ned])
    np.save(ep_dir / "recovery_trajectory.npy", recovery_ned)
    np.save(ep_dir / "recovery_trajectory_mocap.npy", recovery_pos_mocap)

    if recovery_figs_data is not None and "error" not in recovery_figs_data:
        rec_states = recovery_figs_data["Xro"]
        rec_pos_mocap = np.array([convert_from_ned_to_zup(s[:3], perm) for s in rec_states])
        np.savez(
            ep_dir / "recovery_figs.npz",
            Tro=recovery_figs_data["Tro"],
            Xro=recovery_figs_data["Xro"],
            Uro=recovery_figs_data["Uro"],
            positions_mocap=rec_pos_mocap,
        )

    metadata["recovery"] = {
        "feasible": bool(recovery_result.feasible),
        "planning_time_s": float(recovery_result.planning_time_s),
        "num_waypoints": int(len(recovery_ned)),
        "start_step": int(recovery_result.safe_state.step_index),
        "perturbation": perturb_info,
        "validation": recovery_result.metadata,
    }
    with open(ep_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    manifest_path = results_dir / "visualization_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest.get("episodes", []):
            if entry.get("episode_id") == episode_id:
                entry["recovery_trajectory_mocap"] = str(ep_dir / "recovery_trajectory_mocap.npy")
                if recovery_figs_data is not None and "error" not in recovery_figs_data:
                    entry["recovery_figs_mocap"] = str(ep_dir / "recovery_figs.npz")
                break
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def _clear_recovery_outputs(results_dir: Path, episode_id: int) -> None:
    ep_dir = results_dir / "episodes" / f"episode_{episode_id:05d}"
    for name in ("recovery_trajectory.npy", "recovery_trajectory_mocap.npy", "recovery_figs.npz"):
        path = ep_dir / name
        if path.exists():
            path.unlink()

    metadata_path = ep_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        metadata.pop("recovery", None)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    manifest_path = results_dir / "visualization_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest.get("episodes", []):
            if entry.get("episode_id") == episode_id:
                entry.pop("recovery_trajectory_mocap", None)
                entry.pop("recovery_figs_mocap", None)
                break
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--skip-rollout", action="store_true", help="Plan recovery only; do not roll it out in FiGS")
    parser.add_argument("--coarse-search-step", type=int, default=50, help="Backward search stride when the recorded last safe step is infeasible")
    parser.add_argument("--fine-search-step", type=int, default=5, help="Refinement stride after a feasible start is found")
    args = parser.parse_args()

    results_dir = args.results_dir
    ep_dir = results_dir / "episodes" / f"episode_{args.episode:05d}"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {ep_dir}")

    with open(results_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gate = cfg["scene"]["scene_key"]
    cfg = apply_gate_preset(load_config(None), gate)
    with open(results_dir / "config.yaml", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    for section, vals in user_cfg.items():
        if section in cfg and isinstance(cfg[section], dict):
            cfg[section].update(vals)
        else:
            cfg[section] = vals

    metadata, times, states, controls = _load_episode_state(ep_dir)
    failure_record = _build_failure_record(metadata, times, states, controls)
    seed = int(metadata["seed"])

    suite = build_perturbation_suite(cfg["perturbations"])
    suite.reset_all(seed)

    t_figs_to_nerf = _build_figs_to_nerf_transform(
        cfg["scene"]["scene_key"],
        cfg["simulation"]["permutation"],
        config_yml_path=cfg["scene"].get("config_yml"),
    )

    print(f"Planning recovery for episode {args.episode} in {results_dir}")
    print(f"  gate={gate} seed={seed}")

    recovery, perturb_info = _build_perturbed_recovery(cfg, suite)
    goal_ned = convert_to_ned(cfg["simulation"]["goal_position_zup"], cfg["simulation"]["permutation"])

    start_ned = states[0, :3]
    start_state_ned = states[0, :10]
    perturbed_gate_ned = _compute_perturbed_gate_ned(cfg, suite, t_figs_to_nerf)
    print(f"  start_ned={start_ned.tolist()}")
    print(f"  perturbed_gate_ned={perturbed_gate_ned.tolist()}")
    print(f"  goal_ned={goal_ned.tolist()}")

    recovery_result = recovery.plan_via_gate(start_ned, perturbed_gate_ned, goal_ned,
                                             start_state_figs=start_state_ned)

    if not recovery_result.feasible or recovery_result.trajectory_positions is None:
        _clear_recovery_outputs(results_dir, args.episode)
        summary = {
            "episode": args.episode,
            "feasible": False,
            "perturbation": perturb_info,
            "metadata": recovery_result.metadata,
        }
        print(json.dumps(summary, indent=2))
        return

    recovery_figs_data = None
    if not args.skip_rollout:
        recovery_figs_data = _rollout_recovery(cfg, recovery_result, suite)

    _save_outputs(
        results_dir=results_dir,
        episode_id=args.episode,
        metadata=metadata,
        cfg=cfg,
        recovery_result=recovery_result,
        recovery_figs_data=recovery_figs_data,
        perturb_info=perturb_info,
    )

    summary = {
        "episode": args.episode,
        "feasible": True,
        "planning_time_s": recovery_result.planning_time_s,
        "num_recovery_points": len(recovery_result.trajectory_positions),
        "start_step": int(recovery_result.safe_state.step_index),
        "start_ned": recovery_result.safe_state.state[:3].tolist(),
        "goal_ned": goal_ned.tolist(),
        "perturbation": perturb_info,
        "saved": {
            "recovery_trajectory_npy": str(ep_dir / "recovery_trajectory.npy"),
            "recovery_trajectory_mocap_npy": str(ep_dir / "recovery_trajectory_mocap.npy"),
            "recovery_figs_npz": None if recovery_figs_data is None or "error" in recovery_figs_data else str(ep_dir / "recovery_figs.npz"),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
