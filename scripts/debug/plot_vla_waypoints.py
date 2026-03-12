#!/usr/bin/env python3
"""Visualize VLA waypoints from a falsification episode.

Loads waypoints.npz and trajectory.npz from an episode directory,
and plots:
  - Actual drone trajectory (NED or MOCAP)
  - VLA waypoints at each query step (fanned out as lookahead)
  - Gate and goal positions

Usage:
    python scripts/debug/plot_vla_waypoints.py falsification_results/left_gate_baseline
    python scripts/debug/plot_vla_waypoints.py falsification_results/left_gate_baseline --episode 0 --frame mocap
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vla_falsification.falsification.config import (
    GATE_PRESETS, load_config, apply_gate_preset, convert_to_ned,
)
from vla_falsification.utilities.coordinate_transform import _get_perm_diag


def load_episode(ep_dir: Path):
    traj = np.load(ep_dir / "trajectory.npz", allow_pickle=True)
    wp_path = ep_dir / "waypoints.npz"
    if not wp_path.exists():
        print(f"No waypoints.npz in {ep_dir}. Re-run falsification to generate.")
        sys.exit(1)
    wp = np.load(wp_path, allow_pickle=True)
    return traj, wp


def main():
    parser = argparse.ArgumentParser(description="Plot VLA waypoints")
    parser.add_argument("results_dir", type=Path, help="Falsification results directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode ID")
    parser.add_argument("--frame", choices=["ned", "mocap"], default="mocap",
                        help="Coordinate frame for plotting")
    parser.add_argument("--every-n", type=int, default=1,
                        help="Plot waypoints every N steps (default: every step)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    ep_dir = args.results_dir / "episodes" / f"episode_{args.episode:05d}"
    if not ep_dir.exists():
        print(f"Episode dir not found: {ep_dir}")
        sys.exit(1)

    traj, wp = load_episode(ep_dir)

    # Load config for gate/goal positions
    config_path = args.results_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        gate_zup = np.array(cfg["simulation"]["gate_position_zup"])
        goal_zup = np.array(cfg["simulation"]["goal_position_zup"])
        start_zup = np.array(cfg["simulation"]["start_position_zup"])
        perm = cfg["simulation"]["permutation"]
    else:
        print("No config.yaml found, skipping gate/goal markers")
        gate_zup = goal_zup = start_zup = None
        perm = 5

    P = _get_perm_diag(perm)

    # Extract data
    if args.frame == "mocap":
        traj_pos = traj["positions_mocap"]
        wp_positions = wp["positions_mocap"]
        wp_waypoints = wp["waypoints_mocap"]
        frame_label = "MOCAP Z-up"
        axis_labels = ("X (m)", "Y (m)", "Z (m)")
        gate_pos = gate_zup
        goal_pos = goal_zup
        start_pos = start_zup
    else:
        traj_pos = traj["states"][:, :3]
        wp_positions = wp["positions_ned"]
        wp_waypoints = wp["waypoints_ned"]
        frame_label = "FiGS NED"
        axis_labels = ("N (m)", "E (m)", "D (m)")
        gate_pos = P @ gate_zup if gate_zup is not None else None
        goal_pos = P @ goal_zup if goal_zup is not None else None
        start_pos = P @ start_zup if start_zup is not None else None

    wp_steps = wp["steps"]
    raw_actions = wp["raw_actions"]

    # --- 3D Plot ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Drone trajectory
    ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
            "b-", linewidth=1.5, alpha=0.8, label="Drone trajectory")
    ax.scatter(*traj_pos[0], color="green", s=100, marker="o", label="Start")
    ax.scatter(*traj_pos[-1], color="red", s=100, marker="x", label="End")

    # Gate and goal
    if gate_pos is not None:
        ax.scatter(*gate_pos, color="orange", s=200, marker="D",
                   label="Gate", zorder=5)
    if goal_pos is not None:
        ax.scatter(*goal_pos, color="purple", s=200, marker="*",
                   label="Goal", zorder=5)

    # Waypoints: plot lookahead fans at selected steps
    cmap = plt.cm.viridis
    unique_query_steps = []
    seen_steps = set()

    for i in range(len(wp_steps)):
        s = int(wp_steps[i])
        if s not in seen_steps:
            unique_query_steps.append(i)
            seen_steps.add(s)

    # Only plot at VLA query boundaries (action_index == 0 equivalent)
    plot_indices = unique_query_steps[::args.every_n]

    for idx in plot_indices:
        wps = wp_waypoints[idx]
        pos = wp_positions[idx]
        t_frac = int(wp_steps[idx]) / max(int(wp_steps[-1]), 1)
        color = cmap(t_frac)

        # Draw line from current position through waypoints
        trail = np.vstack([pos[np.newaxis, :], wps])
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                "-", color=color, alpha=0.4, linewidth=0.8)
        ax.scatter(wps[0, 0], wps[0, 1], wps[0, 2],
                   color=color, s=15, alpha=0.6)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(f"VLA Waypoints — Episode {args.episode} ({frame_label})")
    ax.legend(loc="upper left")

    # Make axes equal scale
    all_pts = traj_pos
    mid = all_pts.mean(axis=0)
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.1)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    plt.tight_layout()

    # --- 2D subplots: per-axis waypoint vs actual ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    times = traj["times"]
    labels_2d = list(axis_labels)

    for dim in range(3):
        ax2 = axes2[dim]
        ax2.plot(times, traj_pos[:, dim], "b-", linewidth=1.2, label="Actual")

        # Plot first waypoint (next target) at each VLA query
        for idx in unique_query_steps:
            wps = wp_waypoints[idx]
            step = int(wp_steps[idx])
            if step < len(times):
                t = times[step]
                ax2.plot(t, wps[0, dim], "r.", markersize=4, alpha=0.5)

        if gate_pos is not None:
            ax2.axhline(gate_pos[dim], color="orange", linestyle="--",
                        alpha=0.5, label="Gate" if dim == 0 else None)
        if goal_pos is not None:
            ax2.axhline(goal_pos[dim], color="purple", linestyle="--",
                        alpha=0.5, label="Goal" if dim == 0 else None)

        ax2.set_ylabel(labels_2d[dim])
        ax2.grid(True, alpha=0.3)

    axes2[0].legend(loc="upper right")
    axes2[0].set_title(f"Per-axis: Actual vs Next Waypoint — Episode {args.episode} ({frame_label})")
    axes2[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    # --- Action magnitude plot ---
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    action_norms = np.linalg.norm(raw_actions[:, :3], axis=1)
    ax3.plot(wp_steps, action_norms, "k-", linewidth=0.8, alpha=0.6)
    ax3.set_xlabel("Control step")
    ax3.set_ylabel("Action delta magnitude (MOCAP m)")
    ax3.set_title(f"VLA Action Magnitudes — Episode {args.episode}")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save.replace(".png", "_3d.png"), dpi=150)
        fig2.savefig(args.save.replace(".png", "_2d.png"), dpi=150)
        fig3.savefig(args.save.replace(".png", "_actions.png"), dpi=150)
        print(f"Saved figures to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
