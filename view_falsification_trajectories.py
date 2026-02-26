#!/usr/bin/env python3
"""
Nerfstudio Viewer with Falsification Trajectory Overlays.

Loads failure and recovery trajectories produced by run_falsification.py
and renders them in the Gaussian splat viewer.

Usage:
    python view_falsification_trajectories.py --results-dir falsification_results/left_gate
    python view_falsification_trajectories.py --results-dir falsification_results/right_gate --episode 3
    python view_falsification_trajectories.py --results-dir falsification_results/left_gate --failures-only

Based on ns_viewer_with_trajectories.py — only the trajectory loading is changed;
the GSplat viewer setup is identical.
"""

import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ.setdefault('CC', 'gcc-11')
os.environ.setdefault('CXX', 'g++-11')

import time
import json
import argparse
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).parent))
from coordinate_transform import create_transformer_for_scene

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState


# Trajectory colors (0-255 RGB for viser 0.1.x)
TRAJECTORY_COLORS = {
    'failure':        (230, 50, 50),    # red
    'failure_safe':   (50, 130, 230),   # blue (safe prefix before failure)
    'recovery_plan':  (25, 200, 100),   # green (SplatNav planned path)
    'recovery_figs':  (150, 50, 230),   # purple (FiGS rollout of recovery)
    'success':        (50, 130, 230),   # blue (completed trajectory)
}


# ===================================================================
# Config-path resolution (same as ns_viewer_with_trajectories.py)
# ===================================================================

_REPO_ROOT = Path(__file__).parent

CONFIG_PATHS = {
    'left_gate': _REPO_ROOT / "left_gate_9_24_2025_COLMAP" / "sagesplat" / "2025-10-06_215922" / "config.yml",
    'right_gate': _REPO_ROOT / "right_gate_9_30_2025_COLMAP" / "sagesplat" / "2025-10-01_103533" / "config.yml",
}


# ===================================================================
# Trajectory loading from falsification results
# ===================================================================

def load_falsification_trajectories(results_dir: Path, episode_ids=None,
                                     failures_only: bool = False):
    """
    Load trajectories from a falsification results directory.

    Returns a list of dicts, one per episode::

        {
            "episode_id": int,
            "success": bool,
            "failure_positions_mocap": (N, 3) or None,
            "last_safe_step": int or None,
            "recovery_positions_mocap": (M, 3) or None,
            "recovery_figs_positions_mocap": (K, 3) or None,
        }
    """
    manifest_path = results_dir / "visualization_manifest.json"
    if not manifest_path.exists():
        print(f"No visualization_manifest.json in {results_dir}")
        return [], None

    with open(manifest_path) as f:
        manifest = json.load(f)

    scene_key = manifest.get("scene_key", "left_gate")
    entries = manifest.get("episodes", [])

    trajectories = []
    for entry in entries:
        eid = entry["episode_id"]
        if episode_ids is not None and eid not in episode_ids:
            continue
        if failures_only and entry.get("success", True):
            continue

        traj = {
            "episode_id": eid,
            "success": entry.get("success", True),
            "failure_positions_mocap": None,
            "last_safe_step": None,
            "recovery_positions_mocap": None,
            "recovery_figs_positions_mocap": None,
        }

        # Failure trajectory
        fpath = entry.get("failure_trajectory_mocap")
        if fpath and Path(fpath).exists():
            data = np.load(fpath)
            traj["failure_positions_mocap"] = data["positions_mocap"]

        # Read last_safe_step from metadata
        ep_dir = Path(fpath).parent if fpath else None
        if ep_dir and (ep_dir / "metadata.json").exists():
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            fail_info = meta.get("failure", {})
            traj["last_safe_step"] = fail_info.get("last_safe_step")

        # Recovery planned trajectory (MOCAP)
        rpath = entry.get("recovery_trajectory_mocap")
        if rpath and Path(rpath).exists():
            traj["recovery_positions_mocap"] = np.load(rpath)

        # Recovery FiGS rollout (MOCAP)
        rfpath = entry.get("recovery_figs_mocap")
        if rfpath and Path(rfpath).exists():
            data = np.load(rfpath)
            traj["recovery_figs_positions_mocap"] = data["positions_mocap"]

        trajectories.append(traj)

    return trajectories, scene_key


# ===================================================================
# Add trajectories to viewer (mirrors ns_viewer_with_trajectories.py
# add_trajectories_to_viewer but reads falsification data)
# ===================================================================

def add_falsification_trajectories(viewer_state: ViewerState, trajectories: list,
                                    scene_name: str, config_path: Path = None,
                                    bbox_interval: int = 50):
    """Overlay failure + recovery trajectories on the GSplat viewer."""
    print(f"\nLoading {len(trajectories)} falsification trajectories for {scene_name}...")

    try:
        transformer = create_transformer_for_scene(scene_name)
    except Exception as e:
        print(f"  Could not load coordinate transformer: {e}")
        transformer = None

    # Dataparser transform (same logic as ns_viewer_with_trajectories.py)
    dataparser_transform = None
    dataparser_scale = 1.0
    if config_path:
        dt_path = config_path.parent / "dataparser_transforms.json"
        if dt_path.exists():
            with open(dt_path) as f:
                dt = json.load(f)
            dataparser_transform = np.array(dt['transform'])
            dataparser_scale = dt['scale']

    server = viewer_state.viser_server

    def _add_spline(name, positions, color, line_width=3.0):
        """Add a spline through positions (Nx3) with given color."""
        if len(positions) < 2:
            return
        server.add_spline_catmull_rom(
            name=name,
            positions=positions,
            color=color,
            line_width=line_width,
            tension=0.5,
        )

    def _add_marker(name, position, color, radius=0.02):
        server.add_icosphere(
            name=name,
            radius=radius,
            color=color,
            position=tuple(position),
        )

    # Drone physical dimensions (metres)
    DRONE_LENGTH = 0.38
    DRONE_WIDTH  = 0.45
    DRONE_HEIGHT = 0.40
    # MOCAP frame is Z-up.  Markers sit on top of the drone, so the
    # body extends mostly downward (negative Z) from the marker point.
    # 50 mm above marker, 350 mm below.
    BBOX_Z_ABOVE = 0.05
    BBOX_Z_BELOW = DRONE_HEIGHT - BBOX_Z_ABOVE  # 0.35

    _box_faces = np.array([
        [0,2,1], [0,3,2],   # top
        [4,5,6], [4,6,7],   # bottom
        [0,1,5], [0,5,4],   # front
        [2,3,7], [2,7,6],   # back
        [0,4,7], [0,7,3],   # left
        [1,2,6], [1,6,5],   # right
    ], dtype=np.uint32)

    def _add_drone_bbox(name, pos_mocap, color=(255, 200, 50), opacity=0.25):
        """Render a semi-transparent wireframe box at a MOCAP position."""
        x, y, z = pos_mocap
        hl, hw = DRONE_LENGTH / 2, DRONE_WIDTH / 2
        z_top = z + BBOX_Z_ABOVE   # Z-up: above = +Z
        z_bot = z - BBOX_Z_BELOW   # Z-up: below = -Z

        corners_mocap = np.array([
            [x - hl, y - hw, z_top],
            [x + hl, y - hw, z_top],
            [x + hl, y + hw, z_top],
            [x - hl, y + hw, z_top],
            [x - hl, y - hw, z_bot],
            [x + hl, y - hw, z_bot],
            [x + hl, y + hw, z_bot],
            [x - hl, y + hw, z_bot],
        ])

        if transformer:
            corners_v = np.array([transformer.mocap_to_colmap_position(c) for c in corners_mocap])
        else:
            corners_v = corners_mocap

        server.add_mesh_simple(
            name=name,
            vertices=corners_v.astype(np.float32),
            faces=_box_faces,
            color=color,
            wireframe=True,
            opacity=opacity,
            side="double",
        )

    for traj in trajectories:
        eid = traj["episode_id"]
        prefix = f"/falsification/ep_{eid:05d}"

        # --- helper to transform MOCAP → viewer coords ---
        def to_viewer(positions_mocap):
            out = []
            for p in positions_mocap:
                if transformer:
                    p_colmap = transformer.mocap_to_colmap_position(p)
                else:
                    p_colmap = p
                out.append(p_colmap)
            return np.array(out)

        # --- Failure trajectory ---
        fpos = traj.get("failure_positions_mocap")
        if fpos is not None and len(fpos) > 1:
            fpos_v = to_viewer(fpos)

            last_safe = traj.get("last_safe_step")
            if last_safe is not None and last_safe > 0 and not traj["success"]:
                safe_end = min(last_safe + 1, len(fpos_v))

                if safe_end > 1:
                    _add_spline(f"{prefix}/safe_prefix",
                                fpos_v[:safe_end],
                                TRAJECTORY_COLORS['failure_safe'], 5.0)
                if safe_end < len(fpos_v):
                    _add_spline(f"{prefix}/failure_suffix",
                                fpos_v[safe_end - 1:],
                                TRAJECTORY_COLORS['failure'], 5.0)
                _add_marker(f"{prefix}/last_safe",
                            fpos_v[safe_end - 1],
                            TRAJECTORY_COLORS['failure_safe'], 0.025)
            else:
                color_key = 'success' if traj["success"] else 'failure'
                _add_spline(f"{prefix}/trajectory",
                            fpos_v, TRAJECTORY_COLORS[color_key], 5.0)

            _add_marker(f"{prefix}/start", fpos_v[0], (0, 255, 0))
            end_color = (255, 0, 0) if not traj["success"] else (50, 130, 230)
            _add_marker(f"{prefix}/end", fpos_v[-1], end_color)

            # Print start/end positions for debugging
            start_mocap = fpos[0]
            end_mocap = fpos[-1]
            start_viewer = fpos_v[0]
            end_viewer = fpos_v[-1]
            print(f"  Episode {eid:5d}:", flush=True)
            print(f"    Start (MOCAP):  [{start_mocap[0]:.4f}, {start_mocap[1]:.4f}, {start_mocap[2]:.4f}]", flush=True)
            print(f"    End   (MOCAP):  [{end_mocap[0]:.4f}, {end_mocap[1]:.4f}, {end_mocap[2]:.4f}]", flush=True)
            print(f"    Start (viewer): [{start_viewer[0]:.4f}, {start_viewer[1]:.4f}, {start_viewer[2]:.4f}]", flush=True)
            print(f"    End   (viewer): [{end_viewer[0]:.4f}, {end_viewer[1]:.4f}, {end_viewer[2]:.4f}]", flush=True)
            print(f"    Points: {len(fpos)}", flush=True)

        # --- Recovery planned path ---
        rpos = traj.get("recovery_positions_mocap")
        if rpos is not None and len(rpos) > 1:
            rpos_v = to_viewer(rpos)
            _add_spline(f"{prefix}/recovery_plan",
                        rpos_v, TRAJECTORY_COLORS['recovery_plan'], 4.0)
            _add_marker(f"{prefix}/recovery_goal",
                        rpos_v[-1], TRAJECTORY_COLORS['recovery_plan'])
            print(f"    Recovery goal (MOCAP):  [{rpos[-1][0]:.4f}, {rpos[-1][1]:.4f}, {rpos[-1][2]:.4f}]", flush=True)
            print(f"    Recovery goal (viewer): [{rpos_v[-1][0]:.4f}, {rpos_v[-1][1]:.4f}, {rpos_v[-1][2]:.4f}]", flush=True)

        # --- Recovery FiGS rollout ---
        rfpos = traj.get("recovery_figs_positions_mocap")
        if rfpos is not None and len(rfpos) > 1:
            rfpos_v = to_viewer(rfpos)
            _add_spline(f"{prefix}/recovery_figs",
                        rfpos_v, TRAJECTORY_COLORS['recovery_figs'], 3.0)

        # --- Drone bounding boxes along trajectory ---
        if fpos is not None and bbox_interval > 0:
            indices = list(range(0, len(fpos), bbox_interval))
            if (len(fpos) - 1) not in indices:
                indices.append(len(fpos) - 1)
            bbox_color = TRAJECTORY_COLORS['success'] if traj["success"] else TRAJECTORY_COLORS['failure']
            for bi, idx in enumerate(indices):
                _add_drone_bbox(f"{prefix}/bbox_{bi:04d}", fpos[idx],
                                color=bbox_color, opacity=0.35)
            print(f"    Bounding boxes: {len(indices)} (every {bbox_interval} steps)", flush=True)

        status = "success" if traj["success"] else "FAILURE"
        has_rec = rpos is not None
        print(f"    Status: {status}"
              + (f" | recovery: {len(rpos)} pts" if has_rec else ""), flush=True)

    print(f"Trajectories loaded.")


# ===================================================================
# Viewer startup (identical to ns_viewer_with_trajectories.py)
# ===================================================================

def start_viewer(config_path: Path, scene_name: str, trajectories: list,
                 websocket_port: int = 7007, bbox_interval: int = 50):
    """Start nerfstudio viewer and overlay falsification trajectories."""
    print(f"Starting viewer for {scene_name}...")
    print(f"  Config: {config_path}")

    config, pipeline, _, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode="test",
    )

    config.vis = "viewer"
    config.viewer.websocket_port = websocket_port

    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    viewer_callback_lock = Lock()

    viewer_state = ViewerState(
        config.viewer,
        log_filename=viewer_log_path,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
        share=False,
        train_lock=viewer_callback_lock,
    )

    print(f"  {viewer_state.viewer_info[0]}")

    config.logging.local_writer.enable = False
    writer.setup_local_writer(
        config.logging,
        max_iter=config.max_num_iterations,
        banner_messages=viewer_state.viewer_info,
    )

    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    viewer_state.update_scene(step=step)

    if trajectories:
        add_falsification_trajectories(
            viewer_state, trajectories, scene_name, config_path,
            bbox_interval=bbox_interval,
        )

    print(f"\nViewer ready! Press Ctrl+C to exit.")
    print(f"  Legend:  blue=safe prefix  red=failure  green=recovery plan  purple=recovery rollout")
    sys.stdout.flush()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="View falsification trajectories in Gaussian splat viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all failure episodes for left gate
  python view_falsification_trajectories.py --results-dir falsification_results/left_gate --failures-only

  # View a specific episode
  python view_falsification_trajectories.py --results-dir falsification_results/left_gate --episode 3

  # Override config path
  python view_falsification_trajectories.py --results-dir falsification_results/right_gate --load-config /path/to/config.yml
""",
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Path to falsification_results/<gate> directory")
    parser.add_argument("--load-config", type=Path, default=None,
                        help="Override nerfstudio config.yml path")
    parser.add_argument("--episode", type=int, nargs='+', default=None,
                        help="Show only specific episode IDs")
    parser.add_argument("--failures-only", action="store_true",
                        help="Only show episodes that failed")
    parser.add_argument("--port", type=int, default=7007,
                        help="Websocket port for viewer")
    parser.add_argument("--bbox-interval", type=int, default=50,
                        help="Show drone bounding box every N steps (0 to disable)")
    args = parser.parse_args()

    # Load trajectories
    trajectories, scene_key = load_falsification_trajectories(
        args.results_dir,
        episode_ids=set(args.episode) if args.episode else None,
        failures_only=args.failures_only,
    )

    if not trajectories:
        print("No trajectories to display.")
        return

    print(f"Loaded {len(trajectories)} episodes for scene '{scene_key}'")

    # Resolve config path
    config_path = args.load_config
    if config_path is None:
        config_path = CONFIG_PATHS.get(scene_key)
        if config_path is None or not config_path.exists():
            print(f"Config not found for '{scene_key}'. Provide --load-config.")
            return

    start_viewer(
        config_path=config_path,
        scene_name=scene_key,
        trajectories=trajectories,
        websocket_port=args.port,
        bbox_interval=args.bbox_interval,
    )


if __name__ == "__main__":
    main()
