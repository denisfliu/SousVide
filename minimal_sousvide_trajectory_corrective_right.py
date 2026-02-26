#!/usr/bin/env python3
"""
Minimal SousVide trajectory generation with optional corrective waypoints for RIGHT GATE.
Based on minimal_sousvide_trajectory_right.py with added corrective waypoint insertion.

With 50% probability, adds one of four corrective waypoints:
- high, left, right, or low
between the start and gate waypoints.
"""

import os
import json
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import io
from PIL import Image
from tqdm import tqdm
from coordinate_transform import create_transformer_for_scene
import cv2

# Set CUDA device to use GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(f"✅ Set CUDA_VISIBLE_DEVICES=0")

# Also set additional CUDA environment variables to ensure GPU usage
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
print(f"✅ Set CUDA_DEVICE_ORDER=PCI_BUS_ID")

# Add the FiGS source directory to Python path
figs_src_path = Path(__file__).parent / "SousVide" / "FiGS" / "src"
if figs_src_path.exists():
    sys.path.insert(0, str(figs_src_path))
    print(f"✅ Added FiGS path: {figs_src_path}")
else:
    print(f"❌ FiGS path not found: {figs_src_path}")

# SousVide imports
try:
    from figs.simulator import Simulator
    from figs.control.vehicle_rate_mpc import VehicleRateMPC
    import figs.utilities.config_helper as ch
    SOUSVIDE_AVAILABLE = True
    print("✅ SousVide imports successful")
    
    # Verify CUDA setup and configure memory management
    try:
        import torch
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Clear any existing CUDA cache
            torch.cuda.empty_cache()
            
            # Set memory fraction to avoid OOM (use 70% of available memory for better stability)
            torch.cuda.set_per_process_memory_fraction(0.7)
            
            # Configure memory allocator to reduce fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            print(f"✅ CUDA available - Using GPU {current_device}: {device_name}")
            
            # Show current memory usage and availability
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            available_memory = total_memory - memory_reserved
            memory_percent_free = (available_memory / total_memory) * 100
            
            print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            print(f"   Total: {total_memory:.2f}GB, Available: {available_memory:.2f}GB ({memory_percent_free:.1f}% free)")
            
            if memory_percent_free < 70:
                print(f"   ⚠️ Warning: Only {memory_percent_free:.1f}% GPU memory available")
            
        else:
            print("⚠️ CUDA not available - falling back to CPU")
    except ImportError:
        print("⚠️ PyTorch not available - cannot verify CUDA setup")
        
except ImportError as e:
    print(f"❌ SousVide imports failed: {e}")
    SOUSVIDE_AVAILABLE = False

# Define corrective waypoints (Z-up coordinates) - RIGHT GATE specific
CORRECTIVE_WAYPOINTS = {
    'high': np.array([0.103599, -0.450861, 1.0363545]),
    'left': np.array([0.12201483, -0.47764557, 1.809603]),
    'right': np.array([-0.305111378, -0.64580810, 1.381885886]),
    'low': np.array([0.389304, -0.173504, 1.486534])
}

# Active corrective waypoints (for testing/production)
ACTIVE_CORRECTIVES = ['high', 'left', 'right', 'low']  # All corrective waypoints active

def convert_to_ned_coordinates(positions, permutation=0):
    """Convert positions from computer vision coordinates (Z-up) to NED coordinates (Z-down).
    
    SousVide uses North-East-Down (NED) aviation convention:
    - +X = North (forward)
    - +Y = East (right) 
    - +Z = Down (positive downward!)
    
    Computer vision (COLMAP/NerfStudio) uses Z-up convention.
    
    Args:
        positions: [x, y, z] position in COLMAP coordinates
        permutation: Which coordinate transformation to apply (0-7)
    """
    positions_array = np.array(positions)
    ned_positions = positions_array.copy()
    
    # Always flip Z (up becomes down)
    z_ned = -positions_array[2]
    
    # Try different X,Y coordinate transformations
    if permutation == 0:
        # Original: X=X, Y=Y
        ned_positions[0] = positions_array[0]   # X = X
        ned_positions[1] = positions_array[1]   # Y = Y
    elif permutation == 1:
        # 90° rotation: X=Y, Y=X
        ned_positions[0] = positions_array[1]   # X = Y
        ned_positions[1] = positions_array[0]   # Y = X
    elif permutation == 2:
        # 180° rotation: X=-X, Y=-Y
        ned_positions[0] = -positions_array[0]  # X = -X
        ned_positions[1] = -positions_array[1]  # Y = -Y
    elif permutation == 3:
        # 270° rotation: X=-Y, Y=X
        ned_positions[0] = -positions_array[1]  # X = -Y
        ned_positions[1] = positions_array[0]   # Y = X
    elif permutation == 4:
        # Mirror X: X=-X, Y=Y
        ned_positions[0] = -positions_array[0]  # X = -X
        ned_positions[1] = positions_array[1]   # Y = Y
    elif permutation == 5:
        # Mirror Y: X=X, Y=-Y
        ned_positions[0] = positions_array[0]   # X = X
        ned_positions[1] = -positions_array[1]  # Y = -Y
    elif permutation == 6:
        # 90° + mirror X: X=-Y, Y=-X
        ned_positions[0] = -positions_array[1]  # X = -Y
        ned_positions[1] = -positions_array[0]  # Y = -X
    elif permutation == 7:
        # 90° + mirror Y: X=Y, Y=-X
        ned_positions[0] = positions_array[1]   # X = Y
        ned_positions[1] = -positions_array[0]  # Y = -X
    
    ned_positions[2] = z_ned
    return ned_positions

def get_permutation_name(permutation):
    """Get descriptive name for coordinate transformation."""
    names = [
        "original",      # X=X, Y=Y
        "rot90_cw",      # X=Y, Y=X
        "rot180",        # X=-X, Y=-Y
        "rot270_cw",     # X=-Y, Y=X
        "mirror_x",      # X=-X, Y=Y
        "mirror_y",      # X=X, Y=-Y
        "rot90_mirror_x", # X=-Y, Y=-X
        "rot90_mirror_y"  # X=Y, Y=-X
    ]
    return names[permutation]

def convert_from_ned_to_standard(positions_ned, permutation=0):
    """Convert positions from NED back to standard Z-up (COLMAP) coordinates.
    
    Args:
        positions_ned: [x, y, z] position in NED coordinates
        permutation: Which coordinate transformation was applied originally (0-7)
    """
    positions_array = np.array(positions_ned)
    x_n, y_n, z_n = positions_array[0], positions_array[1], positions_array[2]
    
    # Invert XY permutation and sign flips
    if permutation == 0:
        x_s, y_s = x_n, y_n
    elif permutation == 1:
        x_s, y_s = y_n, x_n
    elif permutation == 2:
        x_s, y_s = -x_n, -y_n
    elif permutation == 3:
        x_s, y_s = y_n, -x_n
    elif permutation == 4:
        x_s, y_s = -x_n, y_n
    elif permutation == 5:
        x_s, y_s = x_n, -y_n
    elif permutation == 6:
        x_s, y_s = -y_n, -x_n
    elif permutation == 7:
        x_s, y_s = -y_n, x_n
    else:
        x_s, y_s = x_n, y_n
    
    # Invert Z flip (NED z is positive down)
    z_s = -z_n
    
    return np.array([x_s, y_s, z_s])

def load_gripper_overlay():
    """Load gripper mask and region for overlay on downward camera images.
    
    Returns:
        (gripper_region, gripper_mask) tuple or (None, None) if loading fails
    """
    gripper_dir = "/home/jatucker/droneVLA/gripper_overlay"
    mask_path = f"{gripper_dir}/gripper_mask.png"
    region_path = f"{gripper_dir}/gripper_region.png"
    
    try:
        # Load the pre-extracted gripper mask and region (from fifth frame)
        gripper_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gripper_mask is None:
            print(f"⚠️ Failed to load gripper mask from {mask_path}")
            return None, None
        
        # Load the pre-extracted gripper region
        gripper_region_bgr = cv2.imread(region_path, cv2.IMREAD_COLOR)
        if gripper_region_bgr is None:
            print(f"⚠️ Failed to load gripper region from {region_path}")
            return None, None
        
        # Convert BGR to RGB
        gripper_region = cv2.cvtColor(gripper_region_bgr, cv2.COLOR_BGR2RGB)
        
        # Show mask statistics
        y_coords, _ = np.where(gripper_mask > 0)
        if len(y_coords) > 0:
            y_range = f"{y_coords.min()}-{y_coords.max()}"
            print(f"✅ Gripper overlay loaded: mask {gripper_mask.shape}, region {gripper_region.shape}")
            print(f"   Mask position: rows {y_range}")
            print(f"   Non-zero pixels: {np.sum(gripper_region > 0)} ({100*np.sum(gripper_region > 0)/gripper_region.size:.2f}%)")
        
        return gripper_region, gripper_mask
        
    except Exception as e:
        print(f"⚠️ Failed to load gripper overlay: {e}")
        return None, None

def apply_gripper_overlay(img, gripper_region, gripper_mask):
    """Apply gripper overlay to a single downward camera image.
    
    Args:
        img: Downward camera image (numpy array)
        gripper_region: Pre-extracted gripper region
        gripper_mask: Gripper mask for alpha blending
        
    Returns:
        Image with gripper overlay applied
    """
    if gripper_region is None or gripper_mask is None:
        return img
    
    img_h, img_w = img.shape[:2]
    gripper_h, gripper_w = gripper_region.shape[:2]
    
    # Direct overlay when sizes match (expected case: both 256x256)
    if img.shape[:2] == gripper_region.shape[:2]:
        overlaid_img = img.copy()
        
        # Create alpha mask for blending
        alpha = gripper_mask.astype(np.float32) / 255.0
        
        # Apply slight gaussian blur to alpha for softer edges
        alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0.5)
        alpha_3d = np.stack([alpha_blurred] * 3, axis=2)
        
        # Direct pixel-to-pixel overlay
        overlaid_img = (img * (1 - alpha_3d) + gripper_region * alpha_3d).astype(np.uint8)
        
        return overlaid_img
    
    else:
        # Fallback: scale gripper to fit image
        scale = min(img_h / gripper_h, img_w / gripper_w) * 0.8
        new_h = int(gripper_h * scale)
        new_w = int(gripper_w * scale)
        
        gripper_resized = cv2.resize(gripper_region, (new_w, new_h))
        mask_resized = cv2.resize(gripper_mask, (new_w, new_h))
        
        # Center the gripper
        x = (img_w - new_w) // 2
        y = (img_h - new_h) // 2
        
        # Apply overlay using alpha blending
        overlaid_img = img.copy()
        
        # Create alpha mask with softer edges
        alpha = mask_resized.astype(np.float32) / 255.0
        alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0.5)
        alpha_3d = np.stack([alpha_blurred] * 3, axis=2)
        
        # Blend gripper onto image
        roi = overlaid_img[y:y+new_h, x:x+new_w]
        blended = (roi * (1 - alpha_3d) + gripper_resized * alpha_3d).astype(np.uint8)
        overlaid_img[y:y+new_h, x:x+new_w] = blended
        
        return overlaid_img

def create_trajectory_gif(rgb_images, output_path, duration=100, downsample=2):
    """Create an animated GIF from RGB images.
    
    Args:
        rgb_images: Array of RGB images with shape (frames, height, width, channels)
        output_path: Path where to save the GIF
        duration: Duration between frames in milliseconds
        downsample: Factor to downsample frames (e.g., 2 means every 2nd frame)
    """
    try:
        from PIL import Image
        
        # Convert RGB images to PIL Images and downsample
        frames = []
        for i in range(0, rgb_images.shape[0], downsample):
            # Convert from numpy array (uint8) to PIL Image
            img = Image.fromarray(rgb_images[i])
            frames.append(img)
        
        # Save as animated GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0  # Loop forever
            )
            print(f"   📽️ GIF saved: {output_path} ({len(frames)} frames)")
        else:
            print("   ⚠️ No frames to create GIF")
            
    except ImportError:
        print("   ⚠️ PIL/Pillow not available - skipping GIF generation")
        print("   💡 Install with: pip install Pillow")

def load_ground_truth_waypoints():
    """Load ground truth gate and goal positions from files."""
    # Load gate position - RIGHT GATE
    gate_pos = np.array([0.52, -1.02, 1.5])
    
    # Load goal position from first grasp pose
    grasp_data = np.load('/home/jatucker/Splat-MOVER/scripts/grasp_results/stuffed penguin/stuffed penguin_0/stuffed penguin_graspnet.npy')
    first_grasp_matrix = grasp_data[0]  # First grasp pose (4x4 matrix)
    goal_pos = first_grasp_matrix[:3, 3]  # Extract position from transformation matrix
    goal_pos = np.array([1.421417, -0.3320115, 1.5])
    
    return gate_pos, goal_pos

def select_corrective_waypoint(episode_id, probability=0.5):
    """
    With given probability, select a corrective waypoint.
    
    Args:
        episode_id: Episode identifier (used for seeding)
        probability: Probability of including a corrective waypoint
        
    Returns:
        (waypoint_name, waypoint_position) or (None, None)
    """
    # Use episode_id to seed for reproducibility
    rng = np.random.RandomState(episode_id)
    
    # Decide if we include a corrective waypoint
    if rng.random() < probability:
        # Randomly select one of the ACTIVE corrective waypoints
        waypoint_name = rng.choice(ACTIVE_CORRECTIVES)
        waypoint_pos = CORRECTIVE_WAYPOINTS[waypoint_name].copy()
        return waypoint_name, waypoint_pos
    else:
        return None, None

def generate_training_data(simulator, transformer, dataparser_transform, gripper_region, gripper_mask,
                          scene_name="right_gate_9_30_2025_COLMAP/sagesplat/2025-10-01_103533", 
                          permutation=0, perturbation_scale=0.01, episode_id=0, verbose=False,
                          corrective_probability=0.5):
    """
    Generate training data using SousVide with optional corrective waypoints.
    
    Args:
        dataparser_transform: 4x4 transformation matrix from COLMAP to Nerfstudio coordinates
        simulator: Pre-initialized SousVide simulator (to avoid reloading GPU assets)
        scene_name: Name of the gaussian splat scene
        permutation: Coordinate transformation permutation
        perturbation_scale: Scale for goal z-perturbation
        episode_id: Episode identifier
        verbose: Whether to print detailed progress
        corrective_probability: Probability of adding a corrective waypoint (default: 0.5)
        
    Returns:
        Dictionary with training data including RGB images, states, etc.
    """
    
    if not SOUSVIDE_AVAILABLE:
        raise RuntimeError("SousVide is required")
    
    # Load ground truth waypoints from files (only load once)
    if not hasattr(generate_training_data, '_waypoints_cached'):
        generate_training_data._waypoints_cached = load_ground_truth_waypoints()
    gate_pos_gt, goal_pos_gt = generate_training_data._waypoints_cached
    
    if verbose:
        print(f"🎯 Generating episode {episode_id}")
        print(f"   Ground truth gate pos: {gate_pos_gt}")
        print(f"   Ground truth goal pos: {goal_pos_gt}")
    
    # Apply perturbations BEFORE coordinate transformations
    # Gate: NO randomization (use ground truth position)
    gate_pos_perturbed = gate_pos_gt
    
    # Goal: perturb only in z-direction
    goal_perturbation = np.array([0.0, 0.0, np.random.normal(0, perturbation_scale)])
    goal_pos_perturbed = goal_pos_gt + goal_perturbation
    
    # Past-gate waypoint: perturb in small box [0.1, 0.1, 0.1] around base position - RIGHT GATE
    base_past_gate_pos = np.array([1.48, -1.78, 1.6])
    past_gate_perturbation = np.random.uniform(-0.1, 0.1, 3)  # Small box randomization
    past_gate_pos = base_past_gate_pos + past_gate_perturbation
    
    # Start position: perturb in small box [0.1, 0.1, 0.1] around base position
    base_start_pos = np.array([0.104, -0.0219, 1.364])
    start_perturbation = np.random.uniform(-0.1, 0.1, 3)  # Small box randomization
    start_pos_perturbed = base_start_pos + start_perturbation
    
    # Select corrective waypoint (if any)
    corrective_name, corrective_pos = select_corrective_waypoint(episode_id, corrective_probability)
    
    if verbose:
        if corrective_name:
            print(f"   ✅ Adding corrective waypoint: {corrective_name} at {corrective_pos}")
        else:
            print(f"   ⭕ No corrective waypoint for this episode")
    
    # Convert all waypoints to NED coordinates for SousVide
    start_pos_ned = convert_to_ned_coordinates(start_pos_perturbed, permutation)
    gate_pos_ned = convert_to_ned_coordinates(gate_pos_perturbed, permutation)
    past_gate_pos_ned = convert_to_ned_coordinates(past_gate_pos, permutation)
    goal_pos_ned = convert_to_ned_coordinates(goal_pos_perturbed, permutation)
    
    if verbose:
        print(f"   Start (NED coords): {start_pos_ned}")
        print(f"   Gate (NED coords): {gate_pos_ned}")
        if corrective_name:
            corrective_pos_ned = convert_to_ned_coordinates(corrective_pos, permutation)
            print(f"   Corrective-{corrective_name} (NED coords): {corrective_pos_ned}")
        print(f"   Past gate (NED coords): {past_gate_pos_ned}")
        print(f"   Goal (NED coords): {goal_pos_ned}")
    
    # Create course configuration in SousVide format
    # Dynamically build keyframes based on whether we have a corrective waypoint
    # Adjust timing based on whether corrective waypoint is present
    if corrective_name:
        # With corrective: give more time for the maneuvers
        gate_time = 7.0  # More time after corrective
        past_gate_time = 10.0
        goal_time = 14.0
    else:
        # Without corrective: use original timing
        gate_time = 5.0
        past_gate_time = 8.0
        goal_time = 12.0
    
    keyframes = {
        "start": {
            "t": 0.0,
            "fo": [
                [start_pos_ned[0], 0.0],  # x: start position (NED), velocity=0
                [start_pos_ned[1], 0.0],  # y: start position (NED), velocity=0
                [start_pos_ned[2], 0.0],  # z: start position (NED), velocity=0
                [0.0, 0.0]                # yaw: 0 degrees, angular velocity=0
            ]
        }
    }
    
    # Add corrective waypoint if selected (BEFORE gate)
    if corrective_name:
        corrective_pos_ned = convert_to_ned_coordinates(corrective_pos, permutation)
        keyframes["corrective"] = {
            "t": 3.5,  # Between start (0.0) and gate (7.0) - more time for maneuver
            "fo": [
                [corrective_pos_ned[0], None, None, None],
                [corrective_pos_ned[1], None, None, None],
                [corrective_pos_ned[2], None, None, None],
                [0.0, None, None, None]  # yaw: 0 degrees at corrective waypoint
            ]
        }
    
    # Add gate waypoint
    keyframes["gate"] = {
        "t": gate_time,
        "fo": [
            [gate_pos_ned[0], None, None, None],  # x: gate position (NED)
            [gate_pos_ned[1], None, None, None],  # y: gate position (NED)
            [gate_pos_ned[2], None, None, None],  # z: gate position (NED)
            [np.radians(-30.0), None, None, None] # yaw: -30 degrees at gate
        ]
    }
    
    # Add remaining waypoints
    keyframes["past_gate"] = {
        "t": past_gate_time,
        "fo": [
            [past_gate_pos_ned[0], None, None, None],  # x: past gate position (NED)
            [past_gate_pos_ned[1], None, None, None],  # y: past gate position (NED)
            [past_gate_pos_ned[2], None, None, None],  # z: past gate position (NED)
            [0.0, None, None, None]                   # yaw: 0 degrees past gate
        ]
    }
    
    keyframes["goal"] = {
        "t": goal_time,
        "fo": [
            [goal_pos_ned[0], 0.0],  # x: goal position (NED), velocity=0
            [goal_pos_ned[1], 0.0],  # y: goal position (NED), velocity=0
            [goal_pos_ned[2], 0.0],  # z: goal position (NED), velocity=0
            [0.0, 0.0]               # yaw: 0 degrees at goal, angular velocity=0
        ]
    }
    
    course_config = {
        "waypoints": {
            "Nco": 6,
            "keyframes": keyframes
        },
        "forces": None  # No external forces
    }
    
    # Create custom policy configuration with 10 Hz for dataset compatibility
    custom_policy = {
        "plan": {
            "kT": 10.0,
            "use_l2_time": False
        },
        "track": {
            "hz": 10,  # Set to 10 fps to match existing dataset
            "horizon": 40,
            "Qk": [5.0e-1, 5.0e-1, 5.0e-1, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2],
            "Rk": [1e+0, 1e-1, 1e-1, 1e-2],
            "QN": [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2],
            "Ws": [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2],
            "bounds": {
                "lower": [-1.0, -5.0, -5.0, -5.0],
                "upper": [0.0, 5.0, 5.0, 5.0]
            }
        }
    }
    
    # Initialize controller with custom course and 10 Hz policy
    controller = VehicleRateMPC(
        policy=custom_policy,    # Custom policy with 10 Hz frequency
        course=course_config,    # Use our custom course
        frame="carl"             # Carl drone frame
    )
    
    # Initial conditions
    t0 = 0.0
    # Adjust trajectory duration based on whether corrective waypoint is present
    if corrective_name:
        tf = 14.0  # Longer trajectory with corrective waypoint
    else:
        tf = 12.0  # Standard trajectory without corrective
    
    # Initial state: [x, y, z, vx, vy, vz, qx, qy, qz, qw] in NED coordinates
    x0 = np.array([start_pos_ned[0], start_pos_ned[1], start_pos_ned[2], 
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    # Run simulation with dual cameras - forward and downward
    Tro, Xro, Uro, Fro, Rgb_forward, Rgb_downward, Dpt, Aux = simulate_dual_camera(
        simulator, transformer, dataparser_transform, controller, t0, tf, x0, permutation, gripper_region, gripper_mask)
    
    if verbose:
        print(f"✅ Simulation complete: {len(Tro)} timesteps over {tf:.1f}s")
        print(f"   RGB forward images: {Rgb_forward.shape if Rgb_forward is not None else 'None'}")
        print(f"   RGB downward images: {Rgb_downward.shape if Rgb_downward is not None else 'None'}")
        print(f"   States: {Xro.shape}")
        print(f"   Controls: {Uro.shape}")
    
    # Format training data - Convert to DroneVLA format (convert back from NED before saving)
    training_data = format_for_drone_vla(Tro, Xro, Uro, Rgb_forward, Rgb_downward, Dpt, episode_id, permutation, corrective_name)
    
    return training_data

def simulate_dual_camera(simulator, transformer, dataparser_transform, controller, t0, tf, x0, permutation, gripper_region, gripper_mask):
    """
    Custom simulation loop that renders both forward and downward cameras.
    Based on the SousVide simulator but modified for dual camera support.
    
    Args:
        simulator: SousVide simulator
        transformer: Coordinate transformer for MOCAP→COLMAP conversion
        dataparser_transform: 4x4 transformation matrix from COLMAP to Nerfstudio coordinates
        controller: MPC controller
        t0: Start time
        tf: End time
        x0: Initial state
        permutation: Coordinate transformation permutation for NED→MOCAP conversion
        gripper_region: Pre-extracted gripper region for overlay
        gripper_mask: Gripper mask for alpha blending
    """
    import figs.utilities.transform_helper as th
    import figs.utilities.orientation_helper as oh
    import figs.dynamics.quadcopter_specifications as qs
    from figs.dynamics.external_forces import ExternalForces
    
    # Get simulation configuration
    conFiG = simulator.conFiG
    Rout = conFiG["rollout"]
    Spec = qs.generate_specifications(conFiG["frame"])
    fex = ExternalForces(conFiG["forces"])
    
    # Drone specifications  
    nx, nu = Spec["nx"], Spec["nu"]
    m, kt = Spec["m"], Spec["kt"]
    g, Nrtr = Spec["g"], Spec["Nrtr"]
    Tc2b_forward = Spec["Tc2b"]
    rgb_dim, dpt_dim = Spec["rgb_dim"], Spec["dpt_dim"]
    
    # Update camera transforms:
    # Forward camera: use what was previously "downward" - better real-world representation
    Tc2b_forward_base = np.array([
        [ 1.0,  0.0,  0.0,  0.0],     # Same X (right)
        [ 0.0,  0.0,  1.0,  0.0],     # Y becomes Z (forward becomes down)
        [ 0.0, -1.0,  0.0, -0.05],    # Z becomes -Y (down becomes back), slightly offset
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    
    # Scene-specific rotations for RIGHT GATE forward camera
    # NOTE: These are different from the left gate scene!
    
    # No rotations in body frame - rotations will be applied in Nerfstudio frame
    Tc2b_forward = Tc2b_forward_base.copy()
    
    # Scene-specific rotation: +90 degrees about Z-axis, then -90 degrees about Y-axis, then -90 degrees about X-axis
    R_z90 = np.array([
        [0, -1, 0, 0],  # +90 deg rotation about Z
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    R_y90 = np.array([
        [ 0, 0, -1, 0],  # -90 deg rotation about Y
        [ 0, 1,  0, 0],
        [ 1, 0,  0, 0],
        [ 0, 0,  0, 1]
    ])
    R_x90 = np.array([
        [1,  0,  0, 0],  # -90 deg rotation about X
        [0,  0,  1, 0],
        [0, -1,  0, 0],
        [0,  0,  0, 1]
    ])
    R_nerf_forward = R_x90 @ R_y90 @ R_z90  # Apply Z rotation first, then Y, then X
    
    # True downward camera: aligned with +Z in NED (straight down)
    Tc2b_downward_base = np.array([
        [ 1.0,  0.0,  0.0,  0.0],     # X stays right
        [ 0.0, -1.0,  0.0,  0.0],     # Y flipped to point camera down
        [ 0.0,  0.0, -1.0, -0.05],    # Z flipped to look down (+Z direction in NED), slightly offset
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    
    # No rotations in body frame
    Tc2b_downward = Tc2b_downward_base.copy()
    
    # Scene-specific rotation for downward camera: additional rotations
    R_y90_extra = np.array([
        [ 0, 0, -1, 0],  # Additional -90 deg rotation about Y
        [ 0, 1,  0, 0],
        [ 1, 0,  0, 0],
        [ 0, 0,  0, 1]
    ])
    R_x90_extra = np.array([
        [1,  0,  0, 0],  # Additional +90 deg rotation about X
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1]
    ])
    R_y180 = np.array([
        [-1, 0,  0, 0],  # 180 deg rotation about Y
        [ 0, 1,  0, 0],
        [ 0, 0, -1, 0],
        [ 0, 0,  0, 1]
    ])
    R_x90_final = np.array([
        [1,  0,  0, 0],  # Final -90 deg rotation about X
        [0,  0,  1, 0],
        [0, -1,  0, 0],
        [0,  0,  0, 1]
    ])
    R_z90_final = np.array([
        [ 0, 1, 0, 0],  # Final -90 deg rotation about Z
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]
    ])
    # -45 degree rotation about Y
    import math
    cos45 = math.cos(math.radians(-45))
    sin45 = math.sin(math.radians(-45))
    R_y45_final = np.array([
        [ cos45, 0, sin45, 0],  # Additional -45 deg rotation about Y
        [     0, 1,     0, 0],
        [-sin45, 0, cos45, 0],
        [     0, 0,     0, 1]
    ])
    R_nerf_downward = R_y45_final @ R_z90_final @ R_x90_final @ R_y180 @ R_x90_extra @ R_y90_extra @ R_x90 @ R_y90 @ R_z90  # All rotations for downward camera
    
    # Generate cameras for both views
    camera_forward = simulator.gsplat.generate_output_camera(Spec["camera"])
    camera_downward = simulator.gsplat.generate_output_camera(Spec["camera"])
    
    # Simulation parameters
    hz_sim = Rout["frequency"]
    n_sim2ctl = int(hz_sim / controller.hz)
    dt = np.round(tf - t0, 5)
    Nsim = int(dt * hz_sim)
    Nctl = int(dt * controller.hz)
    
    # Initialize arrays
    nw = 6  # Force + Torque
    Tro = np.zeros((Nctl + 1))
    Xro = np.zeros((Nctl + 1, nx))
    Uro = np.zeros((Nctl, nu))
    Wro = np.zeros((Nctl, nw))
    Rgb_forward = np.zeros(((Nctl,) + rgb_dim), dtype=np.uint8)
    Rgb_downward = np.zeros(((Nctl,) + rgb_dim), dtype=np.uint8)
    Dpt = np.zeros(((Nctl,) + dpt_dim), dtype=np.uint8)
    
    # Noise parameters (simplified)
    mu_md, std_md = np.zeros(nx), np.zeros(nx)
    mu_sn, std_sn = np.zeros(nx), np.zeros(nx)
    
    # Initial state
    xcr = x0.copy()
    xpr = x0.copy()
    xsn = x0.copy()
    ucr = np.array([-(m * g) / (Nrtr * kt), 0.0, 0.0, 0.0])
    
    # Main simulation loop
    tau_cr = np.zeros(3)
    for i in range(Nsim):
        tcr = t0 + i / hz_sim
        fcr = fex.get_forces(xcr[0:6], noisy=True)
        pcr = np.hstack((m, kt, fcr))
        fts = np.hstack((fcr, tau_cr))
        
        # Control and rendering loop
        if i % n_sim2ctl == 0:
            # Get body-to-world transformation (in NED frame from simulation)
            Tb2w_ned = th.x_to_T(xcr)
            
            # Extract position and rotation from NED frame
            pos_ned = Tb2w_ned[:3, 3]
            R_ned = Tb2w_ned[:3, :3]
            
            # Convert position from NED to MOCAP frame (standard Z-up coordinates)
            pos_mocap = convert_from_ned_to_standard(pos_ned, permutation)
            
            # Construct the NED→MOCAP transformation matrix
            # NED: +X North, +Y East, +Z Down
            # MOCAP: +X ?, +Y ?, +Z Up (standard vision coordinates)
            # The conversion involves:
            # 1. Z-axis flip (Down → Up): multiply Z by -1
            # 2. XY permutation based on permutation parameter
            
            # Build the coordinate frame transformation matrix
            T_ned_to_mocap = np.eye(4)
            
            # For permutation 5 (mirror_y): x_m = x_n, y_m = -y_n, z_m = -z_n
            # This maps NED basis vectors to MOCAP basis vectors
            if permutation == 0:
                T_ned_to_mocap[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            elif permutation == 5:  # mirror_y (most common for this setup)
                T_ned_to_mocap[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                # Generic handling for other permutations
                # Default to just flipping Z
                T_ned_to_mocap[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            
            # Transform the rotation: R_mocap = T_ned_to_mocap @ R_ned @ T_ned_to_mocap.T
            # This transforms the rotation matrix from NED frame to MOCAP frame
            R_mocap = T_ned_to_mocap[:3, :3] @ R_ned @ T_ned_to_mocap[:3, :3].T
            
            # Reconstruct Tb2w in MOCAP frame
            Tb2w_mocap = np.eye(4)
            Tb2w_mocap[:3, :3] = R_mocap
            Tb2w_mocap[:3, 3] = pos_mocap
            
            # Compute camera poses in MOCAP frame
            Tc2w_forward_mocap = Tb2w_mocap @ Tc2b_forward
            Tc2w_downward_mocap = Tb2w_mocap @ Tc2b_downward
            
            # Transform to COLMAP frame
            Tc2w_forward_colmap = transformer.mocap_to_colmap_pose(Tc2w_forward_mocap)
            Tc2w_downward_colmap = transformer.mocap_to_colmap_pose(Tc2w_downward_mocap)
            
            # Transform from COLMAP to Nerfstudio using dataparser transform
            # This applies the same transformation that was used during model training
            Tc2w_forward_nerf = dataparser_transform @ Tc2w_forward_colmap
            Tc2w_downward_nerf = dataparser_transform @ Tc2w_downward_colmap
            
            # Apply scene-specific rotations in Nerfstudio frame
            Tc2w_forward_nerf = Tc2w_forward_nerf @ R_nerf_forward
            Tc2w_downward_nerf = Tc2w_downward_nerf @ R_nerf_downward
            
            # Render with Nerfstudio poses
            rgb_forward, dpt_forward = simulator.gsplat.render_rgb(camera_forward, Tc2w_forward_nerf)
            rgb_downward, dpt_downward = simulator.gsplat.render_rgb(camera_downward, Tc2w_downward_nerf)
            
            # Store original size for later restoration
            original_h, original_w = rgb_downward.shape[:2]
            
            # Resize downward image to 256x256 to match gripper size before applying overlay
            rgb_downward_resized = cv2.resize(rgb_downward, (256, 256))
            
            # Apply gripper overlay to downward camera image
            rgb_downward_with_gripper = apply_gripper_overlay(rgb_downward_resized, gripper_region, gripper_mask)
            
            # Resize back to original size for storage
            rgb_downward = cv2.resize(rgb_downward_with_gripper, (original_w, original_h))
            
            # Add sensor noise
            xsn = xcr + np.random.normal(loc=mu_sn, scale=std_sn)
            xsn[6:10] = oh.obedient_quaternion(xsn[6:10], xpr[6:10])
            
            # Generate control command
            ucr, tsol = controller.control(tcr, xsn, ucr, rgb_forward, dpt_forward, fts)
            
            # Log data
            k = i // n_sim2ctl
            if k < len(Tro):
                Tro[k] = tcr
                Xro[k, :] = xcr
                Uro[k, :] = ucr
                Wro[k, 0:3] = fcr
                Rgb_forward[k, :, :, :] = rgb_forward
                Rgb_downward[k, :, :, :] = rgb_downward
                Dpt[k, :, :, :] = dpt_forward  # Use forward camera depth
        
        # Update previous state
        xpr = xcr
        
        # Simulate dynamics
        xcr = simulator.solver.simulate(x=xcr, u=ucr, p=pcr)
        
        # Add model noise
        xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
        xcr[6:10] = oh.obedient_quaternion(xcr[6:10], xpr[6:10])
    
    # Final state
    Tro[Nctl] = t0 + Nsim / hz_sim
    Xro[Nctl, :] = xcr
    
    return Tro, Xro, Uro, Wro, Rgb_forward, Rgb_downward, Dpt, []

def format_for_drone_vla(times, states, controls, rgb_forward, rgb_downward, depth_images, episode_id=0, permutation=0, corrective_name=None):
    """
    Format simulation data to match DroneVLA dataset structure.
    
    Args:
        times: Array of timestamps
        states: Array of state vectors [x, y, z, vx, vy, vz, qx, qy, qz, qw] (NED coords)
        controls: Array of control inputs
        rgb_forward: Forward-facing camera images
        rgb_downward: Downward-facing camera images  
        depth_images: Depth images
        episode_id: Episode identifier
        corrective_name: Name of corrective waypoint if used (None otherwise)
        
    Returns:
        Pandas DataFrame in DroneVLA format
    """
    # Image and io already imported at top level
    
    # Number of frames (excluding final state-only frame)
    num_frames = len(controls)
    
    # Prepare data arrays
    data = []
    
    for i in range(num_frames):
        # Convert images to PNG bytes (matching reference format)
        def image_to_bytes(img_array):
            if img_array is None:
                return None
            # Ensure uint8 format
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            # Resize to 256x256 to match reference
            img = Image.fromarray(img_array)
            img = img.resize((256, 256))
            # Convert to bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return {'bytes': buf.getvalue()}
        
        # State: convert from NED to standard frame before saving
        # Use position (x,y,z) + yaw from quaternion, pad to 7 elements
        state_vec = np.zeros(7)
        if len(states[i]) >= 10:
            # Extract position and convert to standard frame
            pos_ned = states[i][:3]
            pos_std = convert_from_ned_to_standard(pos_ned, permutation)
            state_vec[:3] = pos_std
            # Extract yaw from quaternion (qx, qy, qz, qw) at indices 6:10
            qx, qy, qz, qw = states[i][6:10]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            state_vec[3] = yaw
            # Remaining elements stay zero for now
        
        # Actions: compute deltas (previous - current) in the standard frame
        action_vec = np.zeros(7)
        if i > 0:
            # Compute position deltas in standard frame
            prev_std = convert_from_ned_to_standard(states[i-1][:3], permutation)
            curr_std = convert_from_ned_to_standard(states[i][:3], permutation)
            action_vec[:3] = prev_std - curr_std
            
            # Compute yaw delta
            if len(states[i]) >= 10:
                # Previous yaw
                qx_prev, qy_prev, qz_prev, qw_prev = states[i-1][6:10]
                yaw_prev = np.arctan2(2.0 * (qw_prev * qz_prev + qx_prev * qy_prev), 
                                     1.0 - 2.0 * (qy_prev * qy_prev + qz_prev * qz_prev))
                # Current yaw
                qx_curr, qy_curr, qz_curr, qw_curr = states[i][6:10]
                yaw_curr = np.arctan2(2.0 * (qw_curr * qz_curr + qx_curr * qy_curr),
                                     1.0 - 2.0 * (qy_curr * qy_curr + qz_curr * qz_curr))
                action_vec[3] = yaw_prev - yaw_curr
        
        # Create black image for 3rd person views
        black_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create row data
        row = {
            'image': image_to_bytes(rgb_forward[i] if rgb_forward is not None else None),
            'wrist_image': image_to_bytes(rgb_downward[i] if rgb_downward is not None else None), 
            '3pov_1': image_to_bytes(black_image),  # All black image
            '3pov_2': image_to_bytes(black_image),  # All black image
            'state': state_vec.astype(np.float32),
            'actions': action_vec.astype(np.float32),
            'timestamp': np.float32(times[i]),
            'frame_index': np.int64(i),
            'episode_index': np.int64(episode_id),
            'index': np.int64(i),
            'task_index': np.int64(0)  # Single task: "fly through the gate and hover over the stuffed penguin"
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add corrective waypoint info as metadata (if used)
    if corrective_name:
        df.attrs['corrective_waypoint'] = corrective_name
    
    return df

def save_training_data_parquet(df, output_dir, episode_id):
    """Save training data in parquet format matching DroneVLA structure."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet file with DroneVLA naming convention
    filename = f"episode_{episode_id:06d}.parquet"
    df.to_parquet(output_path / filename, index=False)
    
    # Get corrective info for display
    corrective_info = ""
    if 'corrective_waypoint' in df.attrs:
        corrective_info = f" [corrective: {df.attrs['corrective_waypoint']}]"
    
    print(f"💾 Episode {episode_id} saved: {output_path / filename} ({len(df)} frames){corrective_info}")
    return output_path / filename

def main():
    """Generate ~30,000 training trajectories with optional corrective waypoints."""
    
    print("🚀 Generating DroneVLA training dataset with corrective waypoints (RIGHT GATE)")
    print("   Task: fly through the RIGHT gate and hover over the stuffed penguin")
    print(f"   Available corrective waypoints: {list(CORRECTIVE_WAYPOINTS.keys())}")
    print(f"   Active corrective waypoints: {ACTIVE_CORRECTIVES}")
    
    # Parameters for data generation
    target_episodes = 100  # Generate 100 demonstrations
    perturbation_scale = 0.01  # 1cm standard deviation
    corrective_probability = 0.5  # 50% chance of corrective waypoint
    output_base_dir = "dronevla_training_data_corrective_right"
    scene_name = "right_gate_9_30_2025_COLMAP/sagesplat/2025-10-01_103533"
    
    # Load and display ground truth waypoints
    gate_pos_gt, goal_pos_gt = load_ground_truth_waypoints()
    print(f"   Ground truth gate position: {gate_pos_gt}")
    print(f"   Ground truth goal position: {goal_pos_gt}")
    print(f"   Start perturbation: xyz ± 0.1m (uniform box)")
    print(f"   Gate perturbation: NONE (ground truth)")
    print(f"   Goal perturbation: z-only ± {perturbation_scale:.3f}m")
    print(f"   Post-gate perturbation: xyz ± 0.1m (uniform box)")
    print(f"   Corrective waypoint probability: {corrective_probability*100:.0f}%")
    
    # Use mirror_y permutation (correct transformation for training data)
    permutation = 5  # mirror_y coordinate transformation
    perm_name = get_permutation_name(permutation)
    print(f"   Using coordinate transformation: {perm_name}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize SousVide simulator ONCE (major performance optimization)
    print("🔧 Initializing SousVide simulator (one-time setup)...")
    gsplats_path = Path("/home/jatucker/data/data/splats/gsplats")
    print(f"   Using GSplats path: {gsplats_path}")
    
    # Create GSplat object using standard SousVide function
    print(f"   Loading GSplat model from: {scene_name}")
    print(f"   This may take 30-60 seconds for large models...")
    from time import time as timer
    start_time = timer()
    gsplat_obj = ch.get_gsplat(scene_name, gsplats_path=gsplats_path)
    load_time = timer() - start_time
    print(f"   ✅ GSplat loaded in {load_time:.1f} seconds")
    
    print("   Initializing Simulator...")
    simulator = Simulator(
        gsplat=gsplat_obj,        # Pass GSplat object directly
        method="eval_single",     # Standard evaluation method
        frame="carl"              # Carl drone frame
    )
    print("✅ Simulator initialized and GSplat loaded on GPU")
    
    # Check GPU memory after loading
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU Memory after loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except:
        pass  # Skip GPU memory check if torch not available
    
    # Create coordinate transformer
    print("\n📐 Initializing coordinate transformer...")
    transformer = create_transformer_for_scene('right_gate')
    print("✅ Coordinate transformer ready")
    
    # Load dataparser transform for COLMAP → Nerfstudio conversion
    print("\n📐 Loading dataparser transform...")
    dataparser_path = gsplats_path / scene_name / "dataparser_transforms.json"
    if not dataparser_path.exists():
        raise FileNotFoundError(f"Dataparser transform not found: {dataparser_path}")
    
    with open(dataparser_path, 'r') as f:
        dataparser_data = json.load(f)
    
    # Convert 3x4 transform to 4x4 by adding [0, 0, 0, 1] as last row
    dataparser_transform = np.eye(4)
    dataparser_transform[:3, :] = np.array(dataparser_data['transform'])
    print(f"✅ Loaded dataparser transform from {dataparser_path.name}")
    print(f"   Scale: {dataparser_data['scale']:.4f}")
    
    # Load gripper overlay for downward camera
    print("\n🦾 Loading gripper overlay...")
    gripper_region, gripper_mask = load_gripper_overlay()
    if gripper_region is not None:
        print("✅ Gripper overlay ready")
    else:
        print("⚠️ Gripper overlay not available - downward images will not have gripper")
    
    successful_episodes = 0
    failed_episodes = 0
    # Track stats only for active correctives
    corrective_stats = {name: 0 for name in ACTIVE_CORRECTIVES}
    corrective_stats['none'] = 0
    
    # Use tqdm for progress tracking
    with tqdm(total=target_episodes, desc="Generating episodes", unit="ep") as pbar:
        for episode_id in range(target_episodes):
            try:
                # Generate training data with corrective waypoints
                training_df = generate_training_data(
                    simulator=simulator,
                    transformer=transformer,
                    dataparser_transform=dataparser_transform,
                    gripper_region=gripper_region,
                    gripper_mask=gripper_mask,
                    scene_name=scene_name,
                    permutation=permutation,
                    perturbation_scale=perturbation_scale,
                    episode_id=episode_id,
                    verbose=(episode_id < 5),  # Only verbose for first 5 episodes
                    corrective_probability=corrective_probability
                )
                
                # Track corrective waypoint usage
                if 'corrective_waypoint' in training_df.attrs:
                    corrective_stats[training_df.attrs['corrective_waypoint']] += 1
                else:
                    corrective_stats['none'] += 1
                
                # Determine chunk directory (1000 episodes per chunk)
                chunk_id = episode_id // 1000
                chunk_dir = Path(output_base_dir) / f"chunk-{chunk_id:03d}"
                
                # Save the episode data
                save_training_data_parquet(training_df, chunk_dir, episode_id)
                
                # Create GIFs for the first episode only for visualization
                if episode_id == 0:
                    try:
                        # Extract RGB data from the DataFrame for GIF creation
                        rgb_forward_frames = []
                        rgb_downward_frames = []
                        
                        for idx, row in training_df.iterrows():
                            # Convert PNG bytes back to images for GIF
                            if row['image'] is not None and 'bytes' in row['image']:
                                img_forward = Image.open(io.BytesIO(row['image']['bytes']))
                                rgb_forward_frames.append(np.array(img_forward))
                            
                            if row['wrist_image'] is not None and 'bytes' in row['wrist_image']:
                                img_downward = Image.open(io.BytesIO(row['wrist_image']['bytes']))
                                rgb_downward_frames.append(np.array(img_downward))
                        
                        if rgb_forward_frames:
                            rgb_forward_array = np.array(rgb_forward_frames)
                            forward_gif_path = chunk_dir / "trajectory_forward_episode_000000.gif"
                            create_trajectory_gif(rgb_forward_array, forward_gif_path, downsample=1)
                            tqdm.write(f"   📽️ Forward camera GIF saved: {forward_gif_path}")
                        
                        if rgb_downward_frames:
                            rgb_downward_array = np.array(rgb_downward_frames)
                            downward_gif_path = chunk_dir / "trajectory_downward_episode_000000.gif"
                            create_trajectory_gif(rgb_downward_array, downward_gif_path, downsample=1)
                            tqdm.write(f"   📽️ Downward camera GIF saved: {downward_gif_path}")
                            
                    except Exception as gif_error:
                        tqdm.write(f"   ⚠️ GIF creation failed: {gif_error}")
                
                successful_episodes += 1
                
                # Clear GPU memory less frequently (every 50 episodes)
                if episode_id % 50 == 0 and episode_id > 0:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                    except ImportError:
                        pass
                
                # Update progress bar with corrective stats
                corrective_pct = (corrective_stats['none'] / (episode_id + 1)) * 100
                pbar.set_postfix({
                    'Success': f"{successful_episodes}/{episode_id+1}",
                    'No-Corrective': f"{100-corrective_pct:.0f}%",
                    'GPU': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if 'torch' in locals() and torch.cuda.is_available() else "N/A"
                })
                pbar.update(1)
                    
            except Exception as e:
                failed_episodes += 1
                if failed_episodes < 10:  # Only show first 10 failures
                    tqdm.write(f"❌ Episode {episode_id} failed: {e}")
                elif failed_episodes == 10:
                    tqdm.write(f"❌ Many episodes failing - suppressing further error messages")
                
                # Clear GPU memory even if episode failed
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                pbar.update(1)
    
    print(f"\n{'='*60}")
    print("🎉 DATASET GENERATION COMPLETE!")
    print(f"   Successful episodes: {successful_episodes}/{target_episodes}")
    print(f"   Failed episodes: {failed_episodes}/{target_episodes}")
    print(f"   Success rate: {100*successful_episodes/target_episodes:.1f}%")
    print(f"\n   Corrective Waypoint Statistics:")
    print(f"   ─────────────────────────────────")
    print(f"   No corrective: {corrective_stats['none']} ({100*corrective_stats['none']/successful_episodes:.1f}%)")
    for name in ACTIVE_CORRECTIVES:
        print(f"   {name:>12s}: {corrective_stats[name]} ({100*corrective_stats[name]/successful_episodes:.1f}%)")
    print(f"\n   Output directory: {output_base_dir}/")
    print(f"   Task: fly through the RIGHT gate and hover over the stuffed penguin")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


