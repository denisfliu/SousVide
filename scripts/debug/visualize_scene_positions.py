#!/usr/bin/env python3
"""Visualize the Gaussian splat scene as a point cloud with gate/goal/start markers.

Shows the scene in MOCAP Z-up frame with:
  - Grey: scene Gaussian means
  - Green sphere: start position
  - Orange sphere: gate position
  - Purple sphere: goal position
  - Red line: drone trajectory (if results dir provided)
  - Blue line: VLA waypoints (if available)

Usage:
    # Scene + markers only
    python scripts/debug/visualize_scene_positions.py --gate left_gate

    # Scene + trajectory from a falsification run
    python scripts/debug/visualize_scene_positions.py --gate left_gate \
        --results-dir falsification_results/left_gate_baseline --episode 0

    # Export PLY files instead of interactive viewer
    python scripts/debug/visualize_scene_positions.py --gate left_gate \
        --results-dir falsification_results/left_gate_baseline --export /tmp/scene_debug
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

from vla_falsification.falsification.config import (
    GATE_PRESETS, load_config, apply_gate_preset,
)
from vla_falsification.utilities.coordinate_transform import (
    create_transformer_for_scene, _get_perm_diag,
)


def load_scene_means_ns(config_yml: Path, max_points: int = 200_000):
    """Load Gaussian means in nerfstudio-internal frame."""
    from figs.render.gsplat import GSplat

    gsplat = GSplat(config_yml)
    model = gsplat.pipeline.model
    means = model.means.data.cpu().numpy()

    # Approximate colors from SH DC component
    C0 = 0.28209479177387814
    features_dc = model.features_dc.data.cpu().numpy()  # (N, 3) or (N, 1, 3)
    if features_dc.ndim == 3:
        features_dc = features_dc[:, 0, :]
    colors = np.clip(features_dc * C0 + 0.5, 0, 1)

    # Subsample if needed
    if len(means) > max_points:
        idx = np.random.default_rng(42).choice(len(means), max_points, replace=False)
        means = means[idx]
        colors = colors[idx]

    return means, colors


def ns_to_mocap(points_ns, transformer, dp_transform, dp_scale):
    """Convert points from nerfstudio-internal frame to MOCAP Z-up.

    Chain: NS -> undo dataparser -> undo axis_flip -> COLMAP -> Sim(3) -> MOCAP.
    """
    Tdp_4x4 = np.eye(4)
    Tdp_4x4[:3, :] = dp_transform
    Tdp_scaled_4x4 = np.diag([dp_scale, dp_scale, dp_scale, 1.0]) @ Tdp_4x4
    axis_flip_4x4 = np.diag([1.0, -1.0, -1.0, 1.0])
    Tw2g = Tdp_scaled_4x4 @ axis_flip_4x4
    Tg2w = np.linalg.inv(Tw2g)

    # NS -> COLMAP
    points_colmap = (Tg2w[:3, :3] @ points_ns.T + Tg2w[:3, 3:4]).T
    # COLMAP -> MOCAP
    points_mocap = (transformer.s * (transformer.R @ points_colmap.T) + transformer.t[:, None]).T
    return points_mocap


def make_sphere_points(center, radius, n=200):
    """Generate points on a sphere surface for visualization."""
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    pts = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ])
    return center + radius * pts


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write colored point cloud to PLY (binary for speed)."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(np.array(points[i], dtype=np.float32).tobytes())
            f.write(np.array(colors[i], dtype=np.uint8).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", required=True, choices=["left_gate", "right_gate"])
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Falsification results dir (for trajectory overlay)")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--export", type=str, default=None,
                        help="Export PLY files to this directory instead of interactive viewer")
    parser.add_argument("--marker-radius", type=float, default=0.03,
                        help="Radius of gate/goal/start marker spheres (MOCAP meters)")
    args = parser.parse_args()

    # Load config
    preset = GATE_PRESETS[args.gate]
    config_yml = Path(preset["config_yml"])
    scene_key = preset["scene_key"]
    perm = preset["permutation"]
    P = _get_perm_diag(perm)

    start_zup = np.array(preset["start_position_zup"])
    gate_zup = np.array(preset["gate_position_zup"])
    goal_zup = np.array(preset["goal_position_zup"])

    print(f"Positions (MOCAP Z-up):")
    print(f"  Start: {start_zup}")
    print(f"  Gate:  {gate_zup}")
    print(f"  Goal:  {goal_zup}")

    # Load coordinate transforms
    transformer = create_transformer_for_scene(scene_key)
    dp_path = config_yml.parent / "dataparser_transforms.json"
    dp = json.loads(dp_path.read_text())
    dp_transform = np.array(dp["transform"])
    dp_scale = float(dp["scale"])

    # Load scene
    print("Loading Gaussian splat scene...")
    scene_ns, scene_colors = load_scene_means_ns(config_yml, args.max_points)
    print(f"  {len(scene_ns)} Gaussians loaded")

    # Convert scene to MOCAP
    print("Converting to MOCAP frame...")
    scene_mocap = ns_to_mocap(scene_ns, transformer, dp_transform, dp_scale)

    # Scene bounds
    pmin = scene_mocap.min(axis=0)
    pmax = scene_mocap.max(axis=0)
    print(f"  Scene bounds (MOCAP): [{pmin}] to [{pmax}]")

    # Build marker spheres
    r = args.marker_radius
    start_sphere = make_sphere_points(start_zup, r)
    gate_sphere = make_sphere_points(gate_zup, r)
    goal_sphere = make_sphere_points(goal_zup, r)

    # Load trajectory if available
    traj_mocap = None
    wp_data = None
    if args.results_dir is not None:
        ep_dir = args.results_dir / "episodes" / f"episode_{args.episode:05d}"
        traj_path = ep_dir / "trajectory.npz"
        if traj_path.exists():
            traj = np.load(traj_path, allow_pickle=True)
            traj_mocap = traj["positions_mocap"]
            print(f"  Loaded trajectory: {len(traj_mocap)} points")

        wp_path = ep_dir / "waypoints.npz"
        if wp_path.exists():
            wp_data = np.load(wp_path, allow_pickle=True)
            print(f"  Loaded waypoints: {len(wp_data['steps'])} entries")

    if args.export:
        # --- PLY export mode ---
        out_dir = Path(args.export)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Scene colors (0-255)
        sc = (scene_colors * 255).astype(np.uint8)

        # Combine everything
        all_pts = [scene_mocap]
        all_colors = [sc]

        # Markers
        n_marker = len(start_sphere)
        all_pts.append(start_sphere)
        all_colors.append(np.tile([0, 200, 0], (n_marker, 1)).astype(np.uint8))
        all_pts.append(gate_sphere)
        all_colors.append(np.tile([255, 165, 0], (n_marker, 1)).astype(np.uint8))
        all_pts.append(goal_sphere)
        all_colors.append(np.tile([160, 32, 240], (n_marker, 1)).astype(np.uint8))

        if traj_mocap is not None:
            all_pts.append(traj_mocap)
            all_colors.append(np.tile([255, 50, 50], (len(traj_mocap), 1)).astype(np.uint8))

        pts = np.vstack(all_pts)
        cols = np.vstack(all_colors)
        write_ply(out_dir / "scene_with_markers.ply", pts, cols)
        print(f"\nExported: {out_dir / 'scene_with_markers.ply'}")
        return

    # --- Interactive Open3D viewer ---
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed. Use --export to save PLY files instead.")
        print("  pip install open3d")
        sys.exit(1)

    geometries = []

    # Scene point cloud
    pcd_scene = o3d.geometry.PointCloud()
    pcd_scene.points = o3d.utility.Vector3dVector(scene_mocap)
    pcd_scene.colors = o3d.utility.Vector3dVector(scene_colors)
    geometries.append(pcd_scene)

    # Start marker (green sphere)
    mesh_start = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    mesh_start.translate(start_zup)
    mesh_start.paint_uniform_color([0, 0.8, 0])
    geometries.append(mesh_start)

    # Gate marker (orange sphere)
    mesh_gate = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    mesh_gate.translate(gate_zup)
    mesh_gate.paint_uniform_color([1, 0.65, 0])
    geometries.append(mesh_gate)

    # Goal marker (purple sphere)
    mesh_goal = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    mesh_goal.translate(goal_zup)
    mesh_goal.paint_uniform_color([0.63, 0.13, 0.94])
    geometries.append(mesh_goal)

    # Gate pass radius (wireframe sphere)
    cfg = load_config(None)
    apply_gate_preset(cfg, args.gate)
    gate_radius = cfg["simulation"]["gate_pass_radius_m"]
    mesh_gate_radius = o3d.geometry.TriangleMesh.create_sphere(radius=gate_radius)
    mesh_gate_radius.translate(gate_zup)
    wire_gate = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_gate_radius)
    wire_gate.paint_uniform_color([1, 0.65, 0])
    geometries.append(wire_gate)

    # Trajectory (red line)
    if traj_mocap is not None and len(traj_mocap) > 1:
        lines = [[i, i + 1] for i in range(len(traj_mocap) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(traj_mocap)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0.2, 0.2])
        geometries.append(line_set)

    # VLA waypoints at each query (cyan fans, every 50th step)
    if wp_data is not None:
        wp_steps = wp_data["steps"]
        wp_mocap = wp_data["waypoints_mocap"]
        wp_pos_mocap = wp_data["positions_mocap"]

        # Show waypoints at every VLA query (action_index=0 implied by step changes)
        seen = set()
        query_indices = []
        for i, s in enumerate(wp_steps):
            s = int(s)
            if s not in seen:
                query_indices.append(i)
                seen.add(s)

        # Show every 2nd query for clarity
        for idx in query_indices[::2]:
            wps = wp_mocap[idx]
            pos = wp_pos_mocap[idx]
            fan_pts = np.vstack([pos[np.newaxis, :], wps])
            fan_lines = [[0, j + 1] for j in range(len(wps))]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(fan_pts)
            ls.lines = o3d.utility.Vector2iVector(fan_lines)
            ls.paint_uniform_color([0, 0.8, 0.8])
            geometries.append(ls)

    # Coordinate axes at origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(axes)

    print("\nOpen3D Viewer:")
    print("  Green sphere = Start")
    print("  Orange sphere = Gate (wireframe = pass radius)")
    print("  Purple sphere = Goal")
    print("  Red line = Drone trajectory")
    print("  Cyan fans = VLA waypoints")
    print("\n  Controls: Left-click drag = rotate, Scroll = zoom, Middle-click = pan")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Scene + Positions ({args.gate}, MOCAP Z-up)",
        width=1400, height=900,
    )


if __name__ == "__main__":
    main()
