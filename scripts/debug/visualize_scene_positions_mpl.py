#!/usr/bin/env python3
"""Visualize scene point cloud + gate/goal/start as matplotlib 3D scatter.

Saves an image that can be viewed inline. For interactive panning, run without --save.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CC", "gcc-11")
os.environ.setdefault("CXX", "g++-11")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "FiGS" / "src"))

from vla_falsification.falsification.config import GATE_PRESETS
from vla_falsification.utilities.coordinate_transform import (
    create_transformer_for_scene, _get_perm_diag,
)


def load_scene(config_yml, max_points=80_000):
    from figs.render.gsplat import GSplat
    gsplat = GSplat(config_yml)
    model = gsplat.pipeline.model
    means = model.means.data.cpu().numpy()
    C0 = 0.28209479177387814
    fdc = model.features_dc.data.cpu().numpy()
    if fdc.ndim == 3:
        fdc = fdc[:, 0, :]
    colors = np.clip(fdc * C0 + 0.5, 0, 1)

    # Filter out far-away Gaussians (keep within 3m of scene center)
    center = np.median(means, axis=0)
    dists = np.linalg.norm(means - center, axis=1)
    mask = dists < np.percentile(dists, 95)
    means, colors = means[mask], colors[mask]

    if len(means) > max_points:
        idx = np.random.default_rng(42).choice(len(means), max_points, replace=False)
        means, colors = means[idx], colors[idx]
    return means, colors


def ns_to_mocap(pts, transformer, dp_transform, dp_scale):
    """NS -> MOCAP Z-up. Includes axis flip."""
    Tdp = np.eye(4)
    Tdp[:3, :] = dp_transform
    Ts = np.diag([dp_scale, dp_scale, dp_scale, 1.0]) @ Tdp
    Af = np.diag([1.0, -1.0, -1.0, 1.0])
    Tw2g = Ts @ Af
    Tg2w = np.linalg.inv(Tw2g)
    colmap = (Tg2w[:3, :3] @ pts.T + Tg2w[:3, 3:4]).T
    mocap = (transformer.s * (transformer.R @ colmap.T) + transformer.t[:, None]).T
    return mocap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, choices=["left_gate", "right_gate"])
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--elev", type=float, default=25)
    parser.add_argument("--azim", type=float, default=-60)
    args = parser.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    preset = GATE_PRESETS[args.gate]
    config_yml = Path(preset["config_yml"])
    scene_key = preset["scene_key"]

    start_zup = np.array(preset["start_position_zup"])
    gate_zup = np.array(preset["gate_position_zup"])
    goal_zup = np.array(preset["goal_position_zup"])

    transformer = create_transformer_for_scene(scene_key)
    dp_path = config_yml.parent / "dataparser_transforms.json"
    dp = json.loads(dp_path.read_text())
    dp_transform = np.array(dp["transform"])
    dp_scale = float(dp["scale"])

    print("Loading scene...")
    scene_ns, scene_colors = load_scene(config_yml, args.max_points)
    scene_mocap = ns_to_mocap(scene_ns, transformer, dp_transform, dp_scale)
    print(f"  {len(scene_mocap)} scene points")

    # Crop to relevant area (around start/gate/goal with padding)
    key_pts = np.array([start_zup, gate_zup, goal_zup])
    bbox_min = key_pts.min(axis=0) - 0.5
    bbox_max = key_pts.max(axis=0) + 0.5
    in_bbox = np.all((scene_mocap >= bbox_min) & (scene_mocap <= bbox_max), axis=1)
    scene_crop = scene_mocap[in_bbox]
    colors_crop = scene_colors[in_bbox]
    print(f"  {len(scene_crop)} points in ROI ({len(scene_mocap) - len(scene_crop)} culled)")

    # Load trajectory
    traj_mocap = None
    if args.results_dir:
        ep_dir = args.results_dir / "episodes" / f"episode_{args.episode:05d}"
        traj_path = ep_dir / "trajectory.npz"
        if traj_path.exists():
            traj_mocap = np.load(traj_path)["positions_mocap"]
            print(f"  Trajectory: {len(traj_mocap)} points")

    # --- Plot ---
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Scene points (subsampled for plotting speed)
    n_plot = min(len(scene_crop), 30_000)
    if len(scene_crop) > n_plot:
        idx = np.random.default_rng(0).choice(len(scene_crop), n_plot, replace=False)
        sp, sc = scene_crop[idx], colors_crop[idx]
    else:
        sp, sc = scene_crop, colors_crop
    ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c=sc, s=0.3, alpha=0.4)

    # Markers
    ax.scatter(*start_zup, c="green", s=200, marker="o", label="Start", zorder=10, edgecolors="black")
    ax.scatter(*gate_zup, c="orange", s=300, marker="D", label="Gate", zorder=10, edgecolors="black")
    ax.scatter(*goal_zup, c="purple", s=300, marker="*", label="Goal", zorder=10, edgecolors="black")

    # Annotate with coordinates
    for name, pos, va in [("Start", start_zup, "bottom"), ("Gate", gate_zup, "top"), ("Goal", goal_zup, "bottom")]:
        ax.text(pos[0], pos[1], pos[2], f"  {name}\n  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
                fontsize=7, va=va)

    # Trajectory
    if traj_mocap is not None:
        ax.plot(traj_mocap[:, 0], traj_mocap[:, 1], traj_mocap[:, 2],
                "r-", linewidth=1.5, alpha=0.8, label="Trajectory")
        ax.scatter(*traj_mocap[0], c="lime", s=80, marker="^", zorder=9, label="Traj start")
        ax.scatter(*traj_mocap[-1], c="red", s=80, marker="v", zorder=9, label="Traj end")

    # Reference line: start -> gate -> goal
    ref = np.array([start_zup, gate_zup, goal_zup])
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", linewidth=1, alpha=0.5, label="Ideal path")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Scene + Positions — {args.gate} (MOCAP Z-up)")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=args.elev, azim=args.azim)

    # Equal aspect
    all_pts = np.vstack([scene_crop, key_pts])
    if traj_mocap is not None:
        all_pts = np.vstack([all_pts, traj_mocap])
    mid = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.3)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
