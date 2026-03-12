#!/usr/bin/env python3
"""Visualize the SplatNav voxel grid to debug collision checking.

Renders the non-navigable voxels as a point cloud alongside the gate position,
start/goal, and optionally a min-snap trajectory.  Outputs an interactive Open3D
window and optionally saves to PLY.

Usage:
    python scripts/debug/visualize_splatnav_voxels.py --gate left_gate
    python scripts/debug/visualize_splatnav_voxels.py --gate left_gate --save voxels.ply
    python scripts/debug/visualize_splatnav_voxels.py --gate left_gate --slice-axis 1 --slice-idx 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports that require PYTHONPATH (run via run.sh or set manually)
# ---------------------------------------------------------------------------
from vla_falsification.falsification.config import (
    GATE_PRESETS, DEFAULT_CONFIG, load_config, apply_gate_preset, convert_to_ned,
)
from vla_falsification.utilities.coordinate_transform import build_figs_to_nerf_transform


def build_voxel(gsplat_path: Path, robot_radius: float, resolution: int, device: torch.device):
    """Build the GSplatVoxel and return it along with timing info."""
    import time

    from splat.splat_utils import GSplatLoader
    from splatplan.spline_utils import SplinePlanner
    from splatplan.splatplan import SplatPlan

    gsplat = GSplatLoader(Path(gsplat_path), device)

    robot_config = {"radius": robot_radius, "vmax": 2.0, "amax": 3.0}
    env_config = {
        "lower_bound": torch.tensor([-0.5, -0.5, -0.5], device=device),
        "upper_bound": torch.tensor([0.5, 0.5, 0.5], device=device),
        "resolution": resolution,
    }
    spline_planner = SplinePlanner(spline_deg=6, N_sec=10, device=device)

    t0 = time.time()
    planner = SplatPlan(gsplat, robot_config, env_config, spline_planner, device)
    t1 = time.time()

    voxel = planner.gsplat_voxel
    print(f"Voxel grid built in {t1 - t0:.2f}s")
    return voxel, planner


def voxel_stats(voxel, gate_nerf: np.ndarray | None = None):
    """Print occupancy statistics."""
    grid = voxel.non_navigable_grid
    total = grid.numel()
    occupied = grid.sum().item()
    print(f"Grid shape:      {tuple(grid.shape)}")
    print(f"Cell size (NS):  {voxel.cell_sizes.cpu().numpy()}")
    print(f"Occupied cells:  {occupied}/{total} ({100*occupied/total:.1f}%)")
    print(f"Free cells:      {total - occupied}/{total} ({100*(total-occupied)/total:.1f}%)")

    if gate_nerf is not None:
        p = torch.tensor(gate_nerf, dtype=torch.float32, device=voxel.device)
        idx = voxel.get_indices(p)
        occ = grid[idx[0], idx[1], idx[2]].item()
        print(f"Gate NS pos:     {gate_nerf}")
        print(f"Gate voxel idx:  {idx.cpu().numpy()}")
        print(f"Gate cell occupied: {occ}")

        # Check neighborhood
        r = 3
        i, j, k = idx[0].item(), idx[1].item(), idx[2].item()
        n = grid.shape[0]
        patch = grid[
            max(0, i-r):min(n, i+r+1),
            max(0, j-r):min(n, j+r+1),
            max(0, k-r):min(n, k+r+1),
        ]
        print(f"Gate neighborhood ({2*r+1}³ patch): {patch.sum().item()}/{patch.numel()} occupied")


def create_visualizations(voxel, gate_nerf=None, start_nerf=None, goal_nerf=None,
                          slice_axis=None, slice_idx=None):
    """Build Open3D geometries for the voxel grid and key positions."""
    import open3d as o3d

    grid = voxel.non_navigable_grid
    centers = voxel.grid_centers  # (R, R, R, 3)

    # Get occupied and free cell centers
    occ_mask = grid
    occ_pts = centers[occ_mask].reshape(-1, 3).cpu().numpy()
    free_pts = centers[~occ_mask].reshape(-1, 3).cpu().numpy()

    # Optional slice
    if slice_axis is not None and slice_idx is not None:
        ax = slice_axis
        r = grid.shape[ax]
        idx = min(max(slice_idx, 0), r - 1)
        # Build a mask for the slice
        slice_mask = torch.zeros_like(grid, dtype=bool)
        if ax == 0:
            slice_mask[idx, :, :] = True
        elif ax == 1:
            slice_mask[:, idx, :] = True
        else:
            slice_mask[:, :, idx] = True
        occ_pts = centers[occ_mask & slice_mask].reshape(-1, 3).cpu().numpy()
        free_pts = centers[~occ_mask & slice_mask].reshape(-1, 3).cpu().numpy()
        print(f"Showing slice: axis={ax}, idx={idx}/{r}")

    geometries = []

    # Occupied voxels (red)
    if len(occ_pts) > 0:
        pcd_occ = o3d.geometry.PointCloud()
        pcd_occ.points = o3d.utility.Vector3dVector(occ_pts)
        pcd_occ.paint_uniform_color([0.8, 0.2, 0.2])
        geometries.append(pcd_occ)
        print(f"Occupied points to render: {len(occ_pts)}")

    # Free voxels (light gray, very small — optional, can be heavy)
    # Skip free voxels by default to keep visualization manageable
    # if len(free_pts) > 0:
    #     pcd_free = o3d.geometry.PointCloud()
    #     pcd_free.points = o3d.utility.Vector3dVector(free_pts)
    #     pcd_free.paint_uniform_color([0.9, 0.9, 0.9])
    #     geometries.append(pcd_free)

    # Key positions as colored spheres
    def add_sphere(pos, color, radius=0.01, label=""):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
        if label:
            print(f"  {label}: {pos}")

    if gate_nerf is not None:
        add_sphere(gate_nerf, [0.0, 1.0, 0.0], radius=0.015, label="Gate (green)")
    if start_nerf is not None:
        add_sphere(start_nerf, [0.0, 0.0, 1.0], radius=0.015, label="Start (blue)")
    if goal_nerf is not None:
        add_sphere(goal_nerf, [1.0, 1.0, 0.0], radius=0.015, label="Goal (yellow)")

    # Coordinate axes at origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(axes)

    # Bounding box
    lb = voxel.lower_bound.cpu().numpy()
    ub = voxel.upper_bound.cpu().numpy()
    bbox_pts = np.array([
        [lb[0], lb[1], lb[2]], [ub[0], lb[1], lb[2]],
        [lb[0], ub[1], lb[2]], [ub[0], ub[1], lb[2]],
        [lb[0], lb[1], ub[2]], [ub[0], lb[1], ub[2]],
        [lb[0], ub[1], ub[2]], [ub[0], ub[1], ub[2]],
    ])
    bbox_lines = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_pts)
    line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(line_set)

    return geometries


def main():
    parser = argparse.ArgumentParser(description="Visualize SplatNav voxel grid")
    parser.add_argument("--gate", type=str, default="left_gate", choices=list(GATE_PRESETS.keys()))
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--robot-radius", type=float, default=0.02,
                        help="Robot radius in NS units (default: 0.02)")
    parser.add_argument("--save", type=str, default=None, help="Save occupied voxels to PLY")
    parser.add_argument("--slice-axis", type=int, default=None, choices=[0, 1, 2],
                        help="Show only one slice of the grid (0=X, 1=Y, 2=Z)")
    parser.add_argument("--slice-idx", type=int, default=None,
                        help="Index of the slice to show")
    parser.add_argument("--no-display", action="store_true", help="Skip Open3D window")
    parser.add_argument("--stats-only", action="store_true", help="Only print stats, no visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scene config (same pipeline as run_falsification.py)
    cfg = load_config(None)
    apply_gate_preset(cfg, args.gate)
    preset = GATE_PRESETS[args.gate]
    scene_key = preset["scene_key"]
    perm = cfg["simulation"]["permutation"]

    gsplat_path = Path(cfg["scene"]["config_yml"])
    print(f"GSplat: {gsplat_path}")

    # Build coordinate transform
    T_figs_to_nerf = build_figs_to_nerf_transform(scene_key, perm, gsplat_path)

    def figs_to_nerf(pos):
        p_h = np.append(np.asarray(pos, dtype=float), 1.0)
        return (T_figs_to_nerf @ p_h)[:3]

    # Key positions: convert Z-up to NED
    gate_ned = convert_to_ned(cfg["simulation"]["gate_position_zup"], perm)
    start_ned = convert_to_ned(cfg["simulation"]["start_position_zup"], perm)
    goal_ned = convert_to_ned(cfg["simulation"]["goal_position_zup"], perm)

    gate_nerf = figs_to_nerf(gate_ned)
    start_nerf = figs_to_nerf(start_ned)
    goal_nerf = figs_to_nerf(goal_ned)

    print(f"Gate  NED={gate_ned} → NS={gate_nerf}")
    print(f"Start NED={start_ned} → NS={start_nerf}")
    print(f"Goal  NED={goal_ned} → NS={goal_nerf}")

    # Build voxel grid
    voxel, planner = build_voxel(gsplat_path, args.robot_radius, args.resolution, device)
    voxel_stats(voxel, gate_nerf)

    if args.stats_only:
        return

    # Save PLY if requested
    if args.save:
        import open3d as o3d
        grid = voxel.non_navigable_grid
        centers = voxel.grid_centers
        occ_pts = centers[grid].reshape(-1, 3).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occ_pts)
        pcd.paint_uniform_color([0.8, 0.2, 0.2])
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"Saved {len(occ_pts)} occupied voxels to {args.save}")

    if args.no_display:
        return

    # Visualize
    import open3d as o3d
    geometries = create_visualizations(
        voxel, gate_nerf, start_nerf, goal_nerf,
        slice_axis=args.slice_axis, slice_idx=args.slice_idx,
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"SplatNav Voxels — {args.gate} (res={args.resolution}, r={args.robot_radius})",
        width=1280, height=720,
    )


if __name__ == "__main__":
    main()
