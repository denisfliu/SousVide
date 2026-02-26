#!/usr/bin/env python3
"""
Main entry point for the VLA falsification pipeline.

Usage::

    python run_falsification.py --gate left_gate                 # left gate
    python run_falsification.py --gate right_gate                # right gate
    python run_falsification.py --gate left_gate --num-episodes 200 --seed 42
    python run_falsification.py --config my_config.yaml          # full override

The script:
1. Loads the Gaussian splat scene for the selected gate
2. Wraps a VLA model (OpenPI server) as a FiGS controller
3. Configures perturbations and safety criteria
4. Runs a falsification campaign (N episodes with varying seeds)
5. For every failure, uses SplatNav to plan a safe recovery trajectory
6. Saves failure + recovery trajectories (in MOCAP frame) for visualization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CC", "gcc-11")
os.environ.setdefault("CXX", "g++-11")

_REPO_ROOT = Path(__file__).parent

acados_lib = _REPO_ROOT / "FiGS" / "acados" / "lib"
if acados_lib.exists():
    os.environ["ACADOS_SOURCE_DIR"] = str(_REPO_ROOT / "FiGS" / "acados")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{acados_lib}:{ld}" if ld else str(acados_lib)
    import ctypes
    for lib_name in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
        ctypes.CDLL(str(acados_lib / lib_name))

figs_src = _REPO_ROOT / "FiGS" / "src"
if figs_src.exists():
    sys.path.insert(0, str(figs_src))

splatnav_src = _REPO_ROOT / "external" / "splatnav"
if splatnav_src.exists():
    sys.path.insert(0, str(splatnav_src))

sousvide_src = _REPO_ROOT / "src"
if sousvide_src.exists():
    sys.path.insert(0, str(sousvide_src))

import figs.utilities.config_helper as ch
from figs.simulator import Simulator

from sousvide.control.vla_policy import VLAPolicy, VLAPolicyConfig
from sousvide.falsification.orchestrator import (
    FalsificationOrchestrator,
    OrchestratorConfig,
    run_campaign,
    summarize_campaign,
)
from sousvide.falsification.perturbations import build_perturbation_suite
from sousvide.falsification.splatnav_recovery import RecoveryConfig, SplatNavRecovery

from coordinate_transform import create_transformer_for_scene

# ===================================================================
# Per-gate scene presets
# ===================================================================

GATE_PRESETS: Dict[str, Dict] = {
    "left_gate": {
        "gsplat_name": "left_gate_9_24_2025_COLMAP/sagesplat/2025-10-06_215922",
        "config_yml": _REPO_ROOT / "left_gate_9_24_2025_COLMAP" / "sagesplat" / "2025-10-06_215922" / "config.yml",
        "scene_key": "left_gate",
        "permutation": 5,
        "start_position_zup": [0.104, -0.0219, 1.364],
        "goal_position_zup": [1.421417, -0.3320115, 1.0],
        "gate_position_zup": [0.804785, 0.79716, 1.5],
        "prompt": "go through the gate on the left and hover over the stuffed animal",
    },
    "right_gate": {
        "gsplat_name": "right_gate_9_30_2025_COLMAP/sagesplat/2025-10-01_103533",
        "config_yml": _REPO_ROOT / "right_gate_9_30_2025_COLMAP" / "sagesplat" / "2025-10-01_103533" / "config.yml",
        "scene_key": "right_gate",
        "permutation": 5,
        "start_position_zup": [0.104, -0.0219, 1.364],
        "goal_position_zup": [1.421417, -0.3320115, 1.0],
        "gate_position_zup": [0.804785, 0.79716, 1.5],
        "prompt": "go through the gate on the right and hover over the stuffed animal",
    },
}


# ===================================================================
# Configuration helpers
# ===================================================================

DEFAULT_CONFIG: Dict = {
    "scene": {
        "gsplat_name": None,            # filled by gate preset
        "config_yml": None,             # filled by gate preset
        "gsplats_path": str(_REPO_ROOT),
        "scene_key": None,
        "splatnav_gsplat_path": None,
    },
    "simulation": {
        "t0": 0.0,
        "tf": 12.0,
        "frame_name": "carl",
        "permutation": 5,
        "goal_position_zup": None,
        "start_position_zup": None,
    },
    "vla": {
        "host": "moraband",
        "port": 8000,
        "prompt": None,
        "hz": 10,
        "actions_per_chunk": 50,
        "action_mapper_type": "position_delta",
        "action_mapper_kwargs": {},
        "image_size": 256,
        "mask_third_person": True,
    },
    "safety": {
        "bounds_lower": [-5.0, -5.0, -5.0],
        "bounds_upper": [5.0, 5.0, 5.0],
        "max_speed": 5.0,
        "max_tilt_deg": 60.0,
        "safe_horizon": 3,
    },
    "perturbations": {
        "action": [
            {"type": "ActionNoise", "std": [0.05, 0.1, 0.1, 0.1]},
        ],
        "observation_image": [
            {"type": "ImageNoise", "std": 10.0},
        ],
        "observation_state": [],
        "observation_camera": [],
        "environment_means": [],
        "environment_scales": [],
        "environment_opacities": [],
    },
    "recovery": {
        "enable": True,
        "robot_radius": 0.05,
        "vmax": 2.0,
        "amax": 3.0,
        "recovery_total_time": 5.0,
    },
    "campaign": {
        "num_episodes": 50,
        "seed_offset": 0,
    },
    "output": {
        "dir": None,                    # filled by gate name
    },
}


def apply_gate_preset(cfg: Dict, gate: str) -> Dict:
    """Overlay a gate preset onto the config, without clobbering user overrides."""
    preset = GATE_PRESETS[gate]
    cfg["scene"]["gsplat_name"] = cfg["scene"]["gsplat_name"] or preset["gsplat_name"]
    cfg["scene"]["config_yml"] = str(preset["config_yml"])
    cfg["scene"]["scene_key"] = cfg["scene"]["scene_key"] or preset["scene_key"]
    cfg["scene"]["splatnav_gsplat_path"] = (
        cfg["scene"]["splatnav_gsplat_path"] or str(preset["config_yml"])
    )
    cfg["simulation"]["permutation"] = preset["permutation"]
    cfg["simulation"]["start_position_zup"] = (
        cfg["simulation"]["start_position_zup"] or preset["start_position_zup"]
    )
    cfg["simulation"]["goal_position_zup"] = (
        cfg["simulation"]["goal_position_zup"] or preset["goal_position_zup"]
    )
    cfg["vla"]["prompt"] = cfg["vla"]["prompt"] or preset["prompt"]
    cfg["output"]["dir"] = cfg["output"]["dir"] or f"falsification_results/{gate}"
    return cfg


def load_config(path: str | None) -> Dict:
    """Deep-copy defaults, then merge a user YAML config on top."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for section, vals in user.items():
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(vals)
            else:
                cfg[section] = vals
    return cfg


def convert_to_ned(pos_zup: np.ndarray, perm: int = 5) -> np.ndarray:
    """Z-up → NED."""
    p = np.array(pos_zup, dtype=float)
    z_ned = -p[2]
    if perm == 0:
        return np.array([p[0], p[1], z_ned])
    elif perm == 5:
        return np.array([p[0], -p[1], z_ned])
    elif perm == 2:
        return np.array([-p[0], -p[1], z_ned])
    else:
        return np.array([p[0], p[1], z_ned])


def convert_from_ned_to_zup(pos_ned: np.ndarray, perm: int = 5) -> np.ndarray:
    """NED → Z-up (MOCAP frame).  Inverse of convert_to_ned."""
    x_n, y_n, z_n = pos_ned[0], pos_ned[1], pos_ned[2]
    if perm == 0:
        return np.array([x_n, y_n, -z_n])
    elif perm == 5:
        return np.array([x_n, -y_n, -z_n])
    elif perm == 2:
        return np.array([-x_n, -y_n, -z_n])
    else:
        return np.array([x_n, y_n, -z_n])


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="VLA Falsification Pipeline")
    parser.add_argument("--gate", type=str, choices=list(GATE_PRESETS.keys()),
                        required=True,
                        help="Which gate environment to run (left_gate / right_gate)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--vla-host", type=str, default=None,
                        help="OpenPI server host (e.g. moraband, manaan)")
    parser.add_argument("--vla-port", type=int, default=None,
                        help="OpenPI server port (default 8000)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Task prompt for the VLA")
    args = parser.parse_args()

    # --- Build config ---
    cfg = load_config(args.config)
    cfg = apply_gate_preset(cfg, args.gate)

    if args.num_episodes is not None:
        cfg["campaign"]["num_episodes"] = args.num_episodes
    if args.seed is not None:
        cfg["campaign"]["seed_offset"] = args.seed
    if args.output_dir is not None:
        cfg["output"]["dir"] = args.output_dir
    if args.no_recovery:
        cfg["recovery"]["enable"] = False
    if args.vla_host is not None:
        cfg["vla"]["host"] = args.vla_host
    if args.vla_port is not None:
        cfg["vla"]["port"] = args.vla_port
    if args.prompt is not None:
        cfg["vla"]["prompt"] = args.prompt

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    perm = cfg["simulation"]["permutation"]
    scene_key = cfg["scene"]["scene_key"]

    # ---- Coordinate conversion ----
    start_ned = convert_to_ned(cfg["simulation"]["start_position_zup"], perm)
    goal_ned = convert_to_ned(cfg["simulation"]["goal_position_zup"], perm)
    x0 = np.concatenate([start_ned, np.zeros(3), np.array([0, 0, 0, 1.0])])

    # ---- Load GSplat & simulator ----
    print(f"Loading Gaussian splat for {args.gate}...")
    config_yml = Path(cfg["scene"]["config_yml"])
    if not config_yml.exists():
        raise FileNotFoundError(f"GSplat config not found: {config_yml}")
    from figs.render.gsplat import GSplat
    saved_cwd = Path.cwd()
    os.chdir(config_yml.parent)
    gsplat_obj = GSplat(config_yml)
    os.chdir(saved_cwd)

    print("Initializing FiGS simulator...")
    simulator = Simulator(
        gsplat=gsplat_obj,
        method="eval_single",
        frame=cfg["simulation"]["frame_name"],
    )

    # ---- Build VLA policy ----
    print("Building VLA policy wrapper...")
    vla_cfg = cfg["vla"]
    vla_config = VLAPolicyConfig(
        host=vla_cfg.get("host", "moraband"),
        port=vla_cfg.get("port", 8000),
        prompt=vla_cfg["prompt"],
        hz=vla_cfg["hz"],
        actions_per_chunk=vla_cfg.get("actions_per_chunk", 50),
        image_size=vla_cfg.get("image_size", 256),
        mask_third_person=vla_cfg.get("mask_third_person", True),
        frame=cfg["simulation"]["frame_name"],
    )
    vla_policy = VLAPolicy(vla_config)

    # ---- Build perturbation suite ----
    print("Building perturbation suite...")
    pert_suite = build_perturbation_suite(cfg["perturbations"])

    # ---- Build SplatNav recovery ----
    splatnav_rec = None
    if cfg["recovery"]["enable"]:
        splatnav_path = cfg["scene"].get("splatnav_gsplat_path")
        if splatnav_path is not None and Path(splatnav_path).exists():
            print(f"Initializing SplatNav recovery planner ({splatnav_path})...")
            rec_cfg = RecoveryConfig(
                robot_radius=cfg["recovery"]["robot_radius"],
                vmax=cfg["recovery"]["vmax"],
                amax=cfg["recovery"]["amax"],
            )
            splatnav_rec = SplatNavRecovery(
                gsplat_path=splatnav_path,
                config=rec_cfg,
            )
        else:
            print("SplatNav config.yml not found; recovery planning disabled.")

    # ---- Build camera transforms ----
    Tc2b_forward_base = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, -0.05],
        [0.0, 0.0, 0.0, 1.0],
    ])
    R_z_180 = np.eye(4); R_z_180[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    R_y_180 = np.eye(4); R_y_180[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    R_y_90 = np.eye(4);  R_y_90[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]])
    Tc2b_forward = Tc2b_forward_base @ R_z_180 @ R_y_180 @ R_y_90

    Tc2b_downward_base = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.05],
        [0.0, 0.0, 0.0, 1.0],
    ])
    R_x_180 = np.eye(4); R_x_180[:3, :3] = np.diag([1.0, -1.0, -1.0])
    R_z_90_d = np.eye(4); R_z_90_d[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    Tc2b_downward = Tc2b_downward_base @ R_x_180 @ R_z_90_d

    # ---- Build orchestrator ----
    orch_cfg = OrchestratorConfig(
        t0=cfg["simulation"]["t0"],
        tf=cfg["simulation"]["tf"],
        frame_name=cfg["simulation"]["frame_name"],
        goal_position=goal_ned.tolist(),
        x0=x0.tolist(),
        Tc2b_forward=Tc2b_forward,
        Tc2b_downward=Tc2b_downward,
        bounds_lower=cfg["safety"]["bounds_lower"],
        bounds_upper=cfg["safety"]["bounds_upper"],
        max_speed=cfg["safety"]["max_speed"],
        max_tilt_deg=cfg["safety"]["max_tilt_deg"],
        safe_horizon=cfg["safety"]["safe_horizon"],
        enable_recovery=cfg["recovery"]["enable"],
        recovery_total_time=cfg["recovery"]["recovery_total_time"],
        permutation=perm,
    )

    orchestrator = FalsificationOrchestrator(
        simulator=simulator,
        vla_policy=vla_policy,
        perturbation_suite=pert_suite,
        config=orch_cfg,
        splatnav_recovery=splatnav_rec,
    )

    # ---- Run campaign ----
    n_episodes = cfg["campaign"]["num_episodes"]
    seed_offset = cfg["campaign"]["seed_offset"]

    print(f"\nFalsification campaign: {n_episodes} episodes | gate={args.gate} | seed={seed_offset}")
    print(f"  Start (NED): {start_ned}  |  Goal (NED): {goal_ned}")
    print(f"  Recovery: {'enabled' if splatnav_rec else 'disabled'}")
    print()

    def progress_cb(i: int, ep):
        status = "OK" if ep.success else f"FAIL ({ep.failure_record.failure_type.name})"
        recovery = ""
        if ep.recovery_result:
            recovery = " | recovery: " + ("feasible" if ep.recovery_result.feasible else "infeasible")
        print(f"  Episode {i:4d}/{n_episodes}: {status} "
              f"({ep.wall_time_s:.1f}s, {len(ep.trajectory)} steps){recovery}")

    episodes = run_campaign(
        orchestrator,
        num_episodes=n_episodes,
        seed_offset=seed_offset,
        progress_callback=progress_cb,
    )

    # ---- Summarize ----
    summary = summarize_campaign(episodes)
    print(f"\n{'='*60}")
    print("FALSIFICATION CAMPAIGN SUMMARY")
    print(f"{'='*60}")
    print(f"  Gate:            {args.gate}")
    print(f"  Episodes:        {summary['total_episodes']}")
    print(f"  Successes:       {summary['successes']}")
    print(f"  Failures:        {summary['failures']} "
          f"({summary['failure_rate']*100:.1f}%)")
    if summary['failures'] > 0:
        print(f"  Avg fail step:   {summary['avg_failure_step']:.1f}")
        print(f"  Recovered:       {summary['recovered']} "
              f"({summary['recovery_rate']*100:.1f}%)")
        print(f"  Failure types:   {summary['failure_types']}")
    print(f"  Avg wall time:   {summary['avg_wall_time_s']:.2f}s")
    print(f"{'='*60}")

    # ---- Save results ----
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Per-episode data — save both NED trajectories and MOCAP-frame
    # positions so the viewer can overlay them on the Gaussian splat.
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    vis_index = []  # for the visualization script

    for ep in episodes:
        ep_dir = episodes_dir / f"episode_{ep.episode_id:05d}"
        ep_dir.mkdir(exist_ok=True)

        # --- Failure trajectory (NED + MOCAP) ---
        traj_states = np.array([s.state for s in ep.trajectory])
        traj_controls = np.array([s.control for s in ep.trajectory])
        traj_times = np.array([s.time for s in ep.trajectory])

        traj_pos_mocap = np.array([
            convert_from_ned_to_zup(s[:3], perm) for s in traj_states
        ])

        np.savez(
            ep_dir / "trajectory.npz",
            times=traj_times,
            states=traj_states,
            controls=traj_controls,
            positions_mocap=traj_pos_mocap,
        )

        # --- Recovery trajectory (if available) ---
        recovery_pos_mocap = None
        if ep.recovery_result is not None and ep.recovery_result.trajectory_positions is not None:
            # recovery_result.trajectory_positions is already in NED (FiGS frame)
            recovery_ned = ep.recovery_result.trajectory_positions
            recovery_pos_mocap = np.array([
                convert_from_ned_to_zup(p, perm) for p in recovery_ned
            ])
            np.save(ep_dir / "recovery_trajectory_mocap.npy", recovery_pos_mocap)
            np.save(ep_dir / "recovery_trajectory.npy", recovery_ned)

        if ep.recovery_figs_data is not None and "error" not in ep.recovery_figs_data:
            rec_states = ep.recovery_figs_data["Xro"]
            rec_pos_mocap = np.array([
                convert_from_ned_to_zup(s[:3], perm) for s in rec_states
            ])
            np.savez(
                ep_dir / "recovery_figs.npz",
                Tro=ep.recovery_figs_data["Tro"],
                Xro=ep.recovery_figs_data["Xro"],
                Uro=ep.recovery_figs_data["Uro"],
                positions_mocap=rec_pos_mocap,
            )

        # --- Metadata ---
        ep_meta = {
            "episode_id": ep.episode_id,
            "seed": ep.seed,
            "success": ep.success,
            "num_steps": len(ep.trajectory),
            "wall_time_s": ep.wall_time_s,
            "gate": args.gate,
        }
        if ep.failure_record is not None:
            ep_meta["failure"] = {
                "type": ep.failure_record.failure_type.name,
                "description": ep.failure_record.description,
                "failure_step": ep.failure_record.failure_step,
                "last_safe_step": ep.failure_record.last_safe_step,
            }
        if ep.recovery_result is not None:
            ep_meta["recovery"] = {
                "feasible": ep.recovery_result.feasible,
                "planning_time_s": ep.recovery_result.planning_time_s,
            }

        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(ep_meta, f, indent=2, default=str)

        # Collect for visualization index
        entry = {
            "episode_id": ep.episode_id,
            "success": ep.success,
            "failure_trajectory_mocap": str(ep_dir / "trajectory.npz"),
        }
        if recovery_pos_mocap is not None:
            entry["recovery_trajectory_mocap"] = str(ep_dir / "recovery_trajectory_mocap.npy")
        if ep.recovery_figs_data is not None and "error" not in ep.recovery_figs_data:
            entry["recovery_figs_mocap"] = str(ep_dir / "recovery_figs.npz")
        vis_index.append(entry)

    # Write visualization index (used by view_falsification_trajectories.py)
    vis_manifest = {
        "gate": args.gate,
        "scene_key": scene_key,
        "permutation": perm,
        "episodes": vis_index,
    }
    with open(output_dir / "visualization_manifest.json", "w") as f:
        json.dump(vis_manifest, f, indent=2)

    print(f"\nResults saved to: {output_dir.resolve()}")
    print(f"Visualize with:  python view_falsification_trajectories.py "
          f"--results-dir {output_dir}")


if __name__ == "__main__":
    main()
