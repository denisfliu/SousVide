#!/usr/bin/env python3
"""Replay a falsification episode and render the forward/downward cameras."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CC", "gcc-11")
os.environ.setdefault("CXX", "g++-11")

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "external" / "FiGS" / "src"))

from vla_falsification.falsification.config import GATE_PRESETS
from vla_falsification.falsification.perturbations import build_perturbation_suite
from vla_falsification.utilities.coordinate_transform import (
    build_camera_transforms,
    create_transformer_for_scene,
    _get_perm_diag,
)

import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th
import figs.dynamics.quadcopter_specifications as qs
from figs.render.gsplat import GSplat


def _apply_environment_perturbations(gsplat: GSplat, perturbation_cfg: dict, seed: int) -> dict:
    suite = build_perturbation_suite(perturbation_cfg)
    suite.reset_all(seed)
    model = gsplat.pipeline.model
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
    return result


def _make_contact_sheet(frames: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not frames:
        raise ValueError("No frames provided for contact sheet.")
    h, w, c = frames[0].shape
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r = i // cols
        cidx = i % cols
        sheet[r * h : (r + 1) * h, cidx * w : (cidx + 1) * w] = frame
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10, help="Render every Nth state")
    parser.add_argument("--fps", type=int, default=10, help="Output video fps")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit rendered frames; 0 means all sampled frames")
    args = parser.parse_args()

    with open(args.results_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ep_dir = args.results_dir / "episodes" / f"episode_{args.episode:05d}"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {ep_dir}")

    with open(ep_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    states = np.load(ep_dir / "trajectory.npz")["states"]
    seed = int(metadata["seed"])
    gate = metadata["gate"]
    frame_name = cfg["simulation"]["frame_name"]

    gsplat = GSplat(Path(cfg["scene"]["config_yml"]))
    frame_dict = ch.get_config(frame_name, "frames")
    spec = qs.generate_specifications(frame_dict)
    camera_fwd = gsplat.generate_output_camera(spec["camera"])
    camera_dwn = gsplat.generate_output_camera(spec["camera"])
    tc2b_forward, tc2b_downward = build_camera_transforms()

    # Coordinate transform chain: NED -> MOCAP -> COLMAP
    perm = cfg["simulation"]["permutation"]
    P = _get_perm_diag(perm)
    transformer = create_transformer_for_scene(cfg["scene"]["scene_key"])

    perturbation_info = _apply_environment_perturbations(gsplat, cfg["perturbations"], seed)

    out_dir = ep_dir / "camera_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(0, len(states), max(1, args.stride)))
    if indices[-1] != len(states) - 1:
        indices.append(len(states) - 1)
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    forward_frames: list[np.ndarray] = []
    downward_frames: list[np.ndarray] = []

    for idx in indices:
        xcr = states[idx]
        # NED body-to-world
        tb2w_ned = th.x_to_T(xcr)
        # NED -> MOCAP: flip y, z on position only
        tb2w_mocap = tb2w_ned.copy()
        tb2w_mocap[:3, 3] = P @ tb2w_ned[:3, 3]
        # Camera-to-world in MOCAP, then MOCAP -> COLMAP
        tc2w_fwd_colmap = transformer.mocap_to_colmap_pose(tb2w_mocap @ tc2b_forward)
        tc2w_dwn_colmap = transformer.mocap_to_colmap_pose(tb2w_mocap @ tc2b_downward)
        # render_rgb applies Tw2g (COLMAP -> nerfstudio) internally
        rgb_fwd, _ = gsplat.render_rgb(camera_fwd, tc2w_fwd_colmap)
        rgb_dwn, _ = gsplat.render_rgb(camera_dwn, tc2w_dwn_colmap)
        forward_frames.append(rgb_fwd)
        downward_frames.append(rgb_dwn)

    imageio.mimwrite(out_dir / "forward.mp4", forward_frames, fps=args.fps)
    imageio.mimwrite(out_dir / "downward.mp4", downward_frames, fps=args.fps)
    imageio.imwrite(out_dir / "forward_contact_sheet.png", _make_contact_sheet(forward_frames[: min(12, len(forward_frames))]))
    imageio.imwrite(out_dir / "downward_contact_sheet.png", _make_contact_sheet(downward_frames[: min(12, len(downward_frames))]))

    summary = {
        "results_dir": str(args.results_dir),
        "episode": args.episode,
        "seed": seed,
        "gate": gate,
        "stride": args.stride,
        "num_rendered_frames": len(indices),
        "perturbation": perturbation_info,
        "outputs": {
            "forward_video": str(out_dir / "forward.mp4"),
            "downward_video": str(out_dir / "downward.mp4"),
            "forward_contact_sheet": str(out_dir / "forward_contact_sheet.png"),
            "downward_contact_sheet": str(out_dir / "downward_contact_sheet.png"),
        },
    }
    with open(out_dir / "render_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
