#!/usr/bin/env python3
"""Generate and export a few gate rigid-transform perturbation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from sousvide.falsification.perturbations import (
    GateRigidTransform,
    GateRigidTransformConfig,
)


def _write_colored_cloud(points: np.ndarray, color: np.ndarray, out_path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.repeat(color.reshape(1, 3), points.shape[0], axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)


def _write_combined_cloud(
    gate_points: np.ndarray, table_points: np.ndarray, out_path: Path
) -> None:
    pcd = o3d.geometry.PointCloud()
    all_pts = np.vstack([gate_points, table_points])
    gate_colors = np.repeat(np.array([[1.0, 0.0, 0.0]]), gate_points.shape[0], axis=0)
    table_colors = np.repeat(np.array([[0.1, 0.6, 1.0]]), table_points.shape[0], axis=0)
    all_colors = np.vstack([gate_colors, table_colors])
    pcd.points = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-ply",
        default="/home/jatucker/Splat-MOVER/scripts/renders/pcd/point_cloud.ply",
        help="Path to full scene point cloud (.ply).",
    )
    parser.add_argument(
        "--gate-mask",
        default="/home/jatucker/SousVide/artifacts/left_gate/left_gate_bottom_mask.npy",
        help="Path to left gate mask (.npy boolean array).",
    )
    parser.add_argument(
        "--table-points",
        default="/home/jatucker/SousVide/artifacts/left_gate/left_table_points.npy",
        help="Path to left table points (.npy N x 3).",
    )
    parser.add_argument(
        "--gate-points",
        default="/home/jatucker/SousVide/artifacts/left_gate/left_gate_bottom_points.npy",
        help="Fallback gate points if scene/mask lengths do not match.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/jatucker/SousVide/artifacts/left_gate/perturbation_samples",
        help="Directory to save perturbed scene samples.",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Base random seed.")
    args = parser.parse_args()

    scene_ply = Path(args.scene_ply).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scene_ply.exists():
        raise FileNotFoundError(f"Scene point cloud not found: {scene_ply}")
    scene_pcd = o3d.io.read_point_cloud(str(scene_ply))
    if scene_pcd.is_empty():
        raise ValueError(f"Scene point cloud is empty: {scene_ply}")
    means_base_np = np.asarray(scene_pcd.points, dtype=np.float32)
    means_base = torch.from_numpy(means_base_np)

    gate_mask = np.load(args.gate_mask).astype(bool).reshape(-1)
    gate_mask_path_for_perturb = args.gate_mask
    if gate_mask.shape[0] != means_base.shape[0]:
        gate_points_fallback = np.asarray(np.load(args.gate_points), dtype=np.float32)
        if gate_points_fallback.ndim != 2 or gate_points_fallback.shape[1] != 3:
            raise ValueError(
                "Gate mask/scene size mismatch and fallback gate points are invalid: "
                f"{gate_points_fallback.shape}"
            )
        means_base = torch.from_numpy(gate_points_fallback.copy())
        gate_mask = np.ones(means_base.shape[0], dtype=bool)
        fallback_mask_path = out_dir / "_fallback_gate_mask.npy"
        np.save(fallback_mask_path, gate_mask)
        gate_mask_path_for_perturb = str(fallback_mask_path)
        print(
            "Warning: scene point count does not match mask; using fallback "
            "gate_points file for perturbation preview."
        )
    gate_indices = np.where(gate_mask)[0]
    table_points = np.asarray(np.load(args.table_points), dtype=np.float64)

    _write_colored_cloud(
        means_base[gate_indices].numpy(),
        np.array([1.0, 0.0, 0.0]),
        out_dir / "baseline_gate_points.ply",
    )
    _write_colored_cloud(
        table_points,
        np.array([0.1, 0.6, 1.0]),
        out_dir / "baseline_table_points.ply",
    )
    _write_combined_cloud(
        means_base[gate_indices].numpy(),
        table_points,
        out_dir / "baseline_gate_table_combined.ply",
    )

    summary = []
    for i in range(args.num_samples):
        seed = args.seed_offset + i
        sample_dir = out_dir / f"sample_{i:02d}_seed_{seed}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        perturb = GateRigidTransform(
            GateRigidTransformConfig(
                gate_mask_path=gate_mask_path_for_perturb,
                gate_points_path=args.gate_points,
                table_points_path=args.table_points,
                max_match_distance_m=0.01,
                max_translation_m=(0.04, 0.04, 0.02),
                yaw_range_deg=(-6.0, 6.0),
                min_translation_m=0.002,
                min_abs_yaw_deg=0.5,
                min_table_clearance_m=0.03,
                max_sampling_tries=120,
                strict=True,
            )
        )
        perturb.reset(np.random.RandomState(seed))
        means_pert = perturb.apply(means_base)

        gate_pts = means_pert[gate_indices].numpy()
        _write_colored_cloud(gate_pts, np.array([1.0, 0.0, 0.0]), sample_dir / "gate_points.ply")
        _write_combined_cloud(gate_pts, table_points, sample_dir / "gate_table_combined.ply")

        trans = perturb._translation.tolist() if perturb._translation is not None else None
        yaw_deg = float(np.degrees(perturb._yaw_rad)) if perturb._yaw_rad is not None else None
        meta = {
            "seed": seed,
            "translation_xyz_m": trans,
            "yaw_deg": yaw_deg,
        }
        with open(sample_dir / "transform.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        summary.append(meta)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {args.num_samples} perturbation samples to: {out_dir}")


if __name__ == "__main__":
    main()
