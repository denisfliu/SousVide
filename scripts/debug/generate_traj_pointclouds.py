#!/usr/bin/env python3
"""Generate point clouds of scene + trajectory in MOCAP and Nerfstudio frames.

Outputs:
  <episode_dir>/debug_pointclouds/scene_traj_mocap.ply
  <episode_dir>/debug_pointclouds/scene_traj_nerf.ply
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from vla_falsification.utilities.coordinate_transform import (
    create_transformer_for_scene,
    _get_perm_diag,
)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write a colored point cloud to PLY (ASCII)."""
    assert points.shape[0] == colors.shape[0]
    n = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(
                f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n"
            )


def load_scene_points(config_yml: Path, max_points: int = 100000) -> np.ndarray:
    """Load Gaussian splat means as scene points (nerfstudio internal frame)."""
    import os
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("CC", "gcc-11")
    os.environ.setdefault("CXX", "g++-11")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "FiGS" / "src"))
    from figs.render.gsplat import GSplat

    gsplat = GSplat(config_yml)
    means = gsplat.pipeline.model.means.data.cpu().numpy()
    if len(means) > max_points:
        idx = np.random.default_rng(42).choice(len(means), max_points, replace=False)
        means = means[idx]
    return means, gsplat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-scene-points", type=int, default=100000)
    args = parser.parse_args()

    import yaml
    with open(args.results_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    ep_dir = args.results_dir / "episodes" / f"episode_{args.episode:05d}"
    data = np.load(ep_dir / "trajectory.npz")
    states = data["states"]
    positions_mocap_saved = data["positions_mocap"]

    scene_key = cfg["scene"]["scene_key"]
    perm = cfg["simulation"]["permutation"]
    config_yml = Path(cfg["scene"]["config_yml"])
    P = _get_perm_diag(perm)

    # Recompute MOCAP positions from NED states for verification
    positions_ned = states[:, :3]
    positions_mocap = np.array([P @ p for p in positions_ned])

    print(f"Trajectory: {len(states)} states")
    print(f"Saved MOCAP positions match recomputed: {np.allclose(positions_mocap, positions_mocap_saved)}")
    if not np.allclose(positions_mocap, positions_mocap_saved):
        print(f"  Max diff: {np.abs(positions_mocap - positions_mocap_saved).max():.6f}")

    # Load coordinate transforms
    transformer = create_transformer_for_scene(scene_key)

    # Load dataparser transform
    dp_path = config_yml.parent / "dataparser_transforms.json"
    with open(dp_path) as f:
        dp = json.load(f)
    dp_transform = np.array(dp["transform"])  # 3x4
    dp_scale = float(dp["scale"])
    axis_flip = np.diag([1.0, -1.0, -1.0])

    print(f"Dataparser scale: {dp_scale}")
    print(f"Dataparser transform:\n{dp_transform}")

    # Load scene (means are in nerfstudio internal frame)
    print("Loading scene...")
    scene_means_ns, gsplat = load_scene_points(config_yml, args.max_scene_points)
    print(f"Scene points: {len(scene_means_ns)}")

    # --- Convert scene means from nerfstudio back to MOCAP ---
    # NS -> undo dp -> undo axis_flip -> COLMAP -> Sim(3) -> Sim3-MOCAP
    Tdp_4x4 = np.eye(4)
    Tdp_4x4[:3, :] = dp_transform
    Tdp_scaled_4x4 = np.diag([dp_scale, dp_scale, dp_scale, 1.0]) @ Tdp_4x4
    axis_flip_4x4 = np.diag([1.0, -1.0, -1.0, 1.0])
    Tw2g = Tdp_scaled_4x4 @ axis_flip_4x4
    Tg2w = np.linalg.inv(Tw2g)  # nerfstudio -> COLMAP

    scene_means_colmap = (Tg2w[:3, :3] @ scene_means_ns.T + Tg2w[:3, 3:4]).T
    # Convert each point: p_mocap = s * R @ p_colmap + t
    scene_means_mocap = (transformer.s * (transformer.R @ scene_means_colmap.T) + transformer.t[:, None]).T

    # --- Convert trajectory to nerfstudio ---
    # MOCAP -> COLMAP -> axis_flip -> dataparser
    traj_colmap = np.array([transformer.mocap_to_colmap_position(p) for p in positions_mocap])
    traj_flipped = (axis_flip @ traj_colmap.T).T
    traj_ns = (dp_scale * (dp_transform[:, :3] @ traj_flipped.T + dp_transform[:, 3:4])).T

    print(f"\nTrajectory start (NED):    {positions_ned[0]}")
    print(f"Trajectory start (MOCAP):  {positions_mocap[0]}")
    print(f"Trajectory start (COLMAP): {traj_colmap[0]}")
    print(f"Trajectory start (flipped):{traj_flipped[0]}")
    print(f"Trajectory start (NS):     {traj_ns[0]}")
    print(f"Trajectory start (NS no flip - buggy): {traj_ns_no_flip[0]}")

    out_dir = ep_dir / "debug_pointclouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- MOCAP frame point cloud ---
    scene_color_mocap = np.full((len(scene_means_mocap), 3), [180, 180, 180], dtype=np.uint8)
    traj_color = np.full((len(positions_mocap), 3), [255, 50, 50], dtype=np.uint8)

    all_pts_mocap = np.vstack([scene_means_mocap, positions_mocap])
    all_colors_mocap = np.vstack([scene_color_mocap, traj_color])
    write_ply(out_dir / "scene_traj_mocap.ply", all_pts_mocap, all_colors_mocap)
    print(f"\nWrote {out_dir / 'scene_traj_mocap.ply'}")

    # --- Nerfstudio frame point cloud (with axis flip - correct) ---
    scene_color_ns = np.full((len(scene_means_ns), 3), [180, 180, 180], dtype=np.uint8)
    traj_color_ns = np.full((len(traj_ns), 3), [255, 50, 50], dtype=np.uint8)

    all_pts_ns = np.vstack([scene_means_ns, traj_ns])
    all_colors_ns = np.vstack([scene_color_ns, traj_color_ns])
    write_ply(out_dir / "scene_traj_nerf.ply", all_pts_ns, all_colors_ns)
    print(f"Wrote {out_dir / 'scene_traj_nerf.ply'}")

    # --- Nerfstudio frame point cloud (WITHOUT axis flip - old buggy way) ---
    traj_color_buggy = np.full((len(traj_ns_no_flip), 3), [50, 50, 255], dtype=np.uint8)

    all_pts_ns_both = np.vstack([scene_means_ns, traj_ns, traj_ns_no_flip])
    all_colors_ns_both = np.vstack([scene_color_ns, traj_color_ns, traj_color_buggy])
    write_ply(out_dir / "scene_traj_nerf_comparison.ply", all_pts_ns_both, all_colors_ns_both)
    print(f"Wrote {out_dir / 'scene_traj_nerf_comparison.ply'} (red=with flip, blue=without flip)")

    # --- Trajectory-only PLYs for easy inspection ---
    write_ply(out_dir / "traj_mocap.ply", positions_mocap, traj_color)
    write_ply(out_dir / "traj_nerf.ply", traj_ns, traj_color_ns)
    print(f"Wrote trajectory-only PLYs")

    print("\nDone! Open the PLY files in MeshLab or similar to verify.")


if __name__ == "__main__":
    main()
