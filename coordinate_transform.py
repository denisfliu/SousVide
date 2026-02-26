#!/usr/bin/env python3
"""
Coordinate transformation utilities for converting between MOCAP and COLMAP frames.

This module provides functions to transform camera poses from MOCAP/NED frame
to COLMAP frame for rendering with Gaussian splats.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple


class CoordinateTransformer:
    """
    Handles coordinate transformations between MOCAP and COLMAP frames.
    
    The Sim(3) transformation from COLMAP to MOCAP is:
        p_mocap = s * R @ p_colmap + t
    
    The inverse (MOCAP to COLMAP) is:
        p_colmap = s_inv * R_inv @ p_mocap + t_inv
    where:
        s_inv = 1/s
        R_inv = R^T
        t_inv = -s_inv * R_inv @ t
    
    For camera poses (4x4 transformation matrices), we apply:
        - Rotation to orientation (no scale)
        - Full Sim(3) to position (scale + rotation + translation)
    """
    
    def __init__(self, alignment_dir: str):
        """
        Initialize transformer from alignment directory.
        
        Args:
            alignment_dir: Path to directory containing colmap_to_mocap_sim3.json
        """
        alignment_path = Path(alignment_dir)
        sim3_path = alignment_path / "colmap_to_mocap_sim3.json"
        
        if not sim3_path.exists():
            raise FileNotFoundError(f"Sim(3) file not found: {sim3_path}")
        
        # Load COLMAP to MOCAP transformation
        with open(sim3_path, 'r') as f:
            sim3 = json.load(f)
        
        self.s = sim3['scale']
        self.R = np.array(sim3['R'])
        self.t = np.array(sim3['t'])
        
        # Compute inverse: MOCAP to COLMAP
        self.s_inv = 1.0 / self.s
        self.R_inv = self.R.T
        self.t_inv = -self.s_inv * (self.R_inv @ self.t)
        
        print(f"✅ Loaded coordinate transformation from {alignment_path.name}")
        print(f"   COLMAP→MOCAP scale: {self.s:.4f}")
        print(f"   MOCAP→COLMAP scale: {self.s_inv:.4f}")
    
    def mocap_to_colmap_position(self, pos_mocap: np.ndarray) -> np.ndarray:
        """
        Transform a 3D position from MOCAP frame to COLMAP frame.
        
        Args:
            pos_mocap: 3D position in MOCAP frame
        
        Returns:
            pos_colmap: 3D position in COLMAP frame
        """
        pos_mocap = np.array(pos_mocap)
        # Apply: pos_colmap = s_inv * (R_inv @ pos_mocap) + t_inv
        pos_colmap = self.s_inv * (self.R_inv @ pos_mocap) + self.t_inv
        return pos_colmap
    
    def mocap_to_colmap_pose(self, T_mocap: np.ndarray) -> np.ndarray:
        """
        Transform a camera pose from MOCAP frame to COLMAP frame.
        
        Args:
            T_mocap: 4x4 camera-to-world transformation matrix in MOCAP frame
        
        Returns:
            T_colmap: 4x4 camera-to-world transformation matrix in COLMAP frame
        """
        T_colmap = np.eye(4)
        
        # Transform rotation: R_colmap = R_inv @ R_mocap
        T_colmap[:3, :3] = self.R_inv @ T_mocap[:3, :3]
        
        # Transform position: t_colmap = s_inv * (R_inv @ t_mocap) + t_inv
        T_colmap[:3, 3] = self.s_inv * (self.R_inv @ T_mocap[:3, 3]) + self.t_inv
        
        return T_colmap
    
    def colmap_to_mocap_pose(self, T_colmap: np.ndarray) -> np.ndarray:
        """
        Transform a camera pose from COLMAP frame to MOCAP frame.
        
        Args:
            T_colmap: 4x4 camera-to-world transformation matrix in COLMAP frame
        
        Returns:
            T_mocap: 4x4 camera-to-world transformation matrix in MOCAP frame
        """
        T_mocap = np.eye(4)
        
        # Transform rotation: R_mocap = R @ R_colmap
        T_mocap[:3, :3] = self.R @ T_colmap[:3, :3]
        
        # Transform position: t_mocap = s * (R @ t_colmap) + t
        T_mocap[:3, 3] = self.s * (self.R @ T_colmap[:3, 3]) + self.t
        
        return T_mocap
    
    def get_transformation_info(self) -> dict:
        """Get transformation parameters for debugging."""
        return {
            'colmap_to_mocap': {
                'scale': self.s,
                'rotation': self.R,
                'translation': self.t
            },
            'mocap_to_colmap': {
                'scale': self.s_inv,
                'rotation': self.R_inv,
                'translation': self.t_inv
            }
        }


def create_transformer_for_scene(scene_name: str) -> CoordinateTransformer:
    """
    Create a CoordinateTransformer for a given scene.
    
    Args:
        scene_name: Scene identifier (e.g., 'left_gate' or 'right_gate')
    
    Returns:
        CoordinateTransformer instance
    """
    _this_dir = Path(__file__).parent

    # Prefer local alignment directories (committed to repo), fall back to
    # the original absolute paths for backward compatibility.
    scene_to_alignment = {
        'left_gate': [
            _this_dir / 'left_gate_9_24_2025_alignment',
            Path('/home/jatucker/data/data/left_gate_9_24_2025_alignment'),
        ],
        'right_gate': [
            _this_dir / 'right_gate_9_30_2025_alignment',
            Path('/home/jatucker/data/data/right_gate_9_30_2025_alignment'),
        ],
    }
    
    if scene_name not in scene_to_alignment:
        raise ValueError(f"Unknown scene: {scene_name}. Valid options: {list(scene_to_alignment.keys())}")
    
    for candidate in scene_to_alignment[scene_name]:
        if (candidate / "colmap_to_mocap_sim3.json").exists():
            return CoordinateTransformer(str(candidate))

    tried = [str(p) for p in scene_to_alignment[scene_name]]
    raise FileNotFoundError(
        f"Alignment data not found for '{scene_name}'. Searched: {tried}"
    )


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Coordinate Transformations ===\n")
    
    # Test left_gate
    print("Testing left_gate transformation:")
    transformer_left = create_transformer_for_scene('left_gate')
    
    # Create a test pose in MOCAP frame
    T_mocap_test = np.eye(4)
    T_mocap_test[:3, 3] = [1.0, -0.5, 0.8]  # Some position in MOCAP
    
    print(f"\nTest pose in MOCAP frame:")
    print(f"Position: {T_mocap_test[:3, 3]}")
    
    # Transform to COLMAP
    T_colmap_test = transformer_left.mocap_to_colmap_pose(T_mocap_test)
    print(f"\nTransformed to COLMAP frame:")
    print(f"Position: {T_colmap_test[:3, 3]}")
    
    # Transform back to verify
    T_mocap_back = transformer_left.colmap_to_mocap_pose(T_colmap_test)
    print(f"\nTransformed back to MOCAP:")
    print(f"Position: {T_mocap_back[:3, 3]}")
    
    # Check roundtrip error
    error = np.linalg.norm(T_mocap_back[:3, 3] - T_mocap_test[:3, 3])
    print(f"\nRoundtrip error: {error:.10f} meters")
    
    if error < 1e-6:
        print("✅ Transformation is reversible!")
    else:
        print("❌ Transformation has numerical issues")
    
    print("\n" + "="*60)
    
    # Test right_gate
    print("\nTesting right_gate transformation:")
    transformer_right = create_transformer_for_scene('right_gate')
    
    T_colmap_test2 = transformer_right.mocap_to_colmap_pose(T_mocap_test)
    print(f"\nSame MOCAP pose transformed for right_gate:")
    print(f"Position in COLMAP: {T_colmap_test2[:3, 3]}")
