#!/usr/bin/env python3
"""
Nerfstudio Viewer with Trajectory Overlays for DroneVLA.

This script extends the standard nerfstudio viewer (ns-viewer) to add
trajectory visualization overlays from parquet files.

Usage:
    conda activate jatucker-dronevla
    python ns_viewer_with_trajectories.py --load-config /path/to/config.yml --scene left_gate
"""

import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import time
import json
import argparse
import sys
from pathlib import Path
from threading import Lock
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
import viser

# Add droneVLA path for coordinate_transform
sys.path.insert(0, str(Path(__file__).parent))
from coordinate_transform import create_transformer_for_scene

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState


# Trajectory type colors - Colorblind-safe palette (Okabe-Ito)
# These colors are distinguishable for people with common types of colorblindness
TRAJECTORY_COLORS = {
    'nominal': (0.00, 0.62, 0.45),   # Bluish Green
    'high': (0.80, 0.40, 0.00),      # Vermillion (safe red alternative)
    'left': (0.00, 0.45, 0.70),      # Blue
    'right': (0.90, 0.62, 0.00),     # Orange
    'low': (0.80, 0.60, 0.70)        # Reddish Purple
}


def load_trajectory_from_parquet(parquet_path):
    """Load trajectory positions from parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
        trajectory = []
        for _, row in df.iterrows():
            state = row['state']  # [x, y, z, yaw, ...]
            trajectory.append(state[:3])  # XYZ only
        return np.array(trajectory)
    except Exception as e:
        print(f"❌ Error loading trajectory from {parquet_path}: {e}")
        return None


def add_trajectories_to_viewer(viewer_state: ViewerState, episodes_json_path: str, 
                                scene_name: str, trajectory_types: list = None,
                                config_path: Path = None, skip_dataparser_transform: bool = False,
                                use_viser_scale: bool = True):
    """
    Add trajectory overlays to an existing nerfstudio viewer.
    
    Args:
        viewer_state: The nerfstudio Viewer object
        episodes_json_path: Path to representative_episodes.json
        scene_name: 'left_gate' or 'right_gate'
        trajectory_types: List of trajectory types to show (None = all)
        config_path: Path to nerfstudio config (for dataparser transform)
    """
    print(f"\n🛤️  Loading trajectories for {scene_name}...")
    
    # Load episode paths
    with open(episodes_json_path, 'r') as f:
        episodes_data = json.load(f)
    
    if scene_name not in episodes_data:
        print(f"❌ No data for scene: {scene_name}")
        return
    
    episodes = episodes_data[scene_name]
    
    if trajectory_types is None:
        trajectory_types = list(episodes.keys())
    
    # Load coordinate transformer (MOCAP → COLMAP)
    try:
        transformer = create_transformer_for_scene(scene_name)
        print(f"   ✅ Loaded MOCAP→COLMAP transformer for {scene_name}")
    except Exception as e:
        print(f"   ⚠️  Could not load coordinate transformer: {e}")
        print(f"   Trajectories may not align correctly")
        transformer = None
    
    # Load nerfstudio dataparser transform (only if needed)
    dataparser_transform = None
    dataparser_scale = 1.0
    if config_path and not skip_dataparser_transform:
        dataparser_transforms_path = config_path.parent / "dataparser_transforms.json"
        if dataparser_transforms_path.exists():
            with open(dataparser_transforms_path, 'r') as f:
                dt = json.load(f)
                dataparser_transform = np.array(dt['transform'])  # 3x4 matrix
                dataparser_scale = dt['scale']
                print(f"   ✅ Loaded dataparser transform (scale: {dataparser_scale:.4f})")
        else:
            print(f"   ⚠️  Dataparser transform not found at {dataparser_transforms_path}")
    
    # Access the viser server from the viewer state
    server = viewer_state.viser_server
    
    # Nerfstudio uses a scale factor for viser coordinates
    VISER_SCALE = 10.0  # VISER_NERFSTUDIO_SCALE_RATIO from nerfstudio
    
    # Debug: Check the scale of the gaussian splat scene
    try:
        model = viewer_state.pipeline.model
        if hasattr(model, 'means'):
            means = model.means.detach().cpu().numpy()
            print(f"   📊 Gaussian splat statistics:")
            print(f"      Number of gaussians: {len(means)}")
            print(f"      Mean position range: [{means.min(axis=0)}, {means.max(axis=0)}]")
            print(f"      Mean center: {means.mean(axis=0)}")
    except Exception as e:
        print(f"   ⚠️  Could not get gaussian statistics: {e}")
    
    for traj_type in trajectory_types:
        if traj_type not in episodes:
            print(f"   ⚠️  No episode found for type: {traj_type}")
            continue
        
        parquet_path = episodes[traj_type]
        print(f"   Loading {traj_type}: {Path(parquet_path).name}")
        
        trajectory = load_trajectory_from_parquet(parquet_path)
        
        if trajectory is None or len(trajectory) == 0:
            continue
        
        # Transform trajectory coordinates
        # Pipeline: MOCAP (from parquet) → COLMAP → Viser
        # Note: We skip the dataparser transform by default because nerfstudio's viewer
        # handles the coordinate system internally
        traj_positions_transformed = []
        
        debug_first = True
        for idx, pos_mocap in enumerate(trajectory):
            # Step 1: MOCAP → COLMAP (using coordinate_transform)
            if transformer:
                pos_colmap = transformer.mocap_to_colmap_position(pos_mocap)
            else:
                pos_colmap = pos_mocap  # Fallback: no transformation
            
            # Step 2: Apply nerfstudio dataparser transform (optional)
            if dataparser_transform is not None and not skip_dataparser_transform:
                # Dataparser transform converts COLMAP to nerfstudio coordinates
                # For world points: p_ns = scale * (R @ p + t)
                pos_transformed = dataparser_transform[:, :3] @ pos_colmap + dataparser_transform[:, 3]
                pos_nerfstudio = pos_transformed * dataparser_scale
            else:
                pos_nerfstudio = pos_colmap
                pos_transformed = None
            
            # Step 3: Apply viser scale (nerfstudio convention) - optional
            if use_viser_scale:
                pos_final = pos_nerfstudio * VISER_SCALE
            else:
                pos_final = pos_nerfstudio
            
            if debug_first and idx == 0:
                print(f"   🔍 First trajectory point:")
                print(f"      MOCAP:          {pos_mocap}")
                print(f"      COLMAP:         {pos_colmap}")
                print(f"      Nerfstudio:     {pos_nerfstudio}")
                if use_viser_scale:
                    print(f"      Viser (10x):    {pos_final}")
                else:
                    print(f"      Final (no 10x): {pos_final}")
                debug_first = False
            
            traj_positions_transformed.append(pos_final)
        
        traj_positions = np.array(traj_positions_transformed)
        
        # Create line segments for trajectory
        points = np.stack([traj_positions[:-1], traj_positions[1:]], axis=1)
        
        # Color trajectory
        color = TRAJECTORY_COLORS.get(traj_type, (0.5, 0.5, 0.5))
        
        # Add trajectory to scene with thicker, more visible lines
        server.scene.add_line_segments(
            name=f"/trajectories/{traj_type}",
            points=points,
            colors=np.array(color),
            line_width=6.0,  # Increased from 3.0 for better visibility
        )
        
        # Add start marker (green sphere) - much smaller
        server.scene.add_icosphere(
            name=f"/trajectories/start_{traj_type}",
            position=traj_positions[0],
            radius=0.02,  # Small fixed size
            color=np.array([0.0, 1.0, 0.0]),
        )
        
        # Add end marker (red sphere) - much smaller
        server.scene.add_icosphere(
            name=f"/trajectories/end_{traj_type}",
            position=traj_positions[-1],
            radius=0.02,  # Small fixed size
            color=np.array([1.0, 0.0, 0.0]),
        )
        
        print(f"   ✅ Added {traj_type} trajectory ({len(traj_positions)} points)")
    
    print(f"✅ Trajectory overlays added!")


def start_viewer_with_trajectories(
    config_path: Path,
    scene_name: str,
    episodes_json_path: Path,
    trajectory_types: list = None,
    websocket_port: int = 7007,
    skip_dataparser_transform: bool = False,
    use_viser_scale: bool = True,
):
    """
    Start nerfstudio viewer with trajectory overlays.
    
    This follows the same pattern as nerfstudio's run_viewer.py but adds
    trajectory overlays after the viewer is initialized.
    """
    print(f"🚀 Starting Nerfstudio Viewer with Trajectory Overlays")
    print(f"   Config: {config_path}")
    print(f"   Scene: {scene_name}")
    
    # Load the model using nerfstudio's standard approach
    print(f"\n📦 Loading model...")
    config, pipeline, _, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode="test",
    )
    
    # Set viewer configuration
    config.vis = "viewer"
    config.viewer.websocket_port = websocket_port
    
    # Start the viewer (same as nerfstudio's _start_viewer)
    print(f"\n🌐 Starting viewer...")
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
    
    print(f"   {viewer_state.viewer_info[0]}")
    
    # Setup logging (required by nerfstudio)
    config.logging.local_writer.enable = False
    writer.setup_local_writer(
        config.logging,
        max_iter=config.max_num_iterations,
        banner_messages=viewer_state.viewer_info
    )
    
    # Initialize the scene
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    viewer_state.update_scene(step=step)
    
    # Add trajectory overlays
    import sys
    sys.stdout.flush()
    
    if episodes_json_path and episodes_json_path.exists():
        add_trajectories_to_viewer(
            viewer_state,
            episodes_json_path,
            scene_name,
            trajectory_types,
            config_path,
            skip_dataparser_transform=skip_dataparser_transform,
            use_viser_scale=use_viser_scale
        )
        sys.stdout.flush()
    else:
        print(f"⚠️  No trajectories to load (episodes file not found: {episodes_json_path})")
        sys.stdout.flush()
    
    print(f"\n✅ Viewer ready! Press Ctrl+C to exit.")
    sys.stdout.flush()
    
    # Keep the viewer running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n👋 Shutting down viewer...")


def main():
    parser = argparse.ArgumentParser(
        description="Nerfstudio viewer with trajectory overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View left gate with all trajectories
  python ns_viewer_with_trajectories.py --scene left_gate
  
  # View right gate with specific trajectories
  python ns_viewer_with_trajectories.py --scene right_gate --trajectory-types nominal high
  
  # View gaussian splat only (no trajectories)
  python ns_viewer_with_trajectories.py --scene left_gate --no-trajectories
"""
    )
    parser.add_argument(
        "--load-config",
        type=Path,
        default=None,
        help="Path to config YAML file (optional, auto-detected from --scene if not provided)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=['left_gate', 'right_gate'],
        required=True,
        help="Scene name for trajectory lookup"
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path(__file__).parent / "representative_episodes.json",
        help="Path to representative_episodes.json"
    )
    parser.add_argument(
        "--trajectory-types",
        nargs='+',
        choices=['nominal', 'high', 'left', 'right', 'low'],
        default=None,
        help="Specific trajectory types to show (default: all)"
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Don't load any trajectories (standard ns-viewer mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7007,
        help="Websocket port for viewer (default: 7007)"
    )
    parser.add_argument(
        "--apply-dataparser-transform",
        action="store_true",
        help="Apply nerfstudio dataparser transform to trajectories (usually not needed)"
    )
    parser.add_argument(
        "--use-viser-scale",
        action="store_true",
        help="Apply 10x viser scale to trajectory coordinates (usually NOT needed)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect config path if not provided
    if args.load_config is None:
        config_paths = {
            'left_gate': Path("/home/jatucker/data/data/splats/gsplats/left_gate_9_24_2025_COLMAP/sagesplat/2025-10-06_215922/config.yml"),
            'right_gate': Path("/home/jatucker/data/data/splats/gsplats/right_gate_9_30_2025_COLMAP/sagesplat/2025-10-01_103533/config.yml"),
        }
        args.load_config = config_paths[args.scene]
        if not args.load_config.exists():
            print(f"❌ Config not found: {args.load_config}")
            print(f"   Please provide --load-config explicitly")
            return
    
    # Don't load trajectories if --no-trajectories is set
    episodes_path = None if args.no_trajectories else args.episodes
    
    start_viewer_with_trajectories(
        config_path=args.load_config,
        scene_name=args.scene,
        episodes_json_path=episodes_path,
        trajectory_types=args.trajectory_types,
        websocket_port=args.port,
        skip_dataparser_transform=not args.apply_dataparser_transform,  # Skip by default
        use_viser_scale=args.use_viser_scale,  # Don't use by default
    )


if __name__ == "__main__":
    main()
