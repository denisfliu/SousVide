"""
Coordinate transformation utilities for the VLA falsification framework.

Handles conversions between:
- FiGS NED (North-East-Down): dynamics simulation frame
- MOCAP Z-up: motion capture reference frame
- COLMAP: camera-centric reconstruction frame
- Nerfstudio-internal: frame used by SplatNav planner

The Sim(3) transformation from COLMAP to MOCAP is:
    p_mocap = s * R @ p_colmap + t
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
# Sim(3) transformer (COLMAP ↔ MOCAP)
# ===================================================================


class CoordinateTransformer:
    """Handles Sim(3) coordinate transformations between MOCAP and COLMAP frames.

    The inverse (MOCAP to COLMAP) is:
        p_colmap = s_inv * R_inv @ p_mocap + t_inv
    where s_inv = 1/s, R_inv = R^T, t_inv = -s_inv * R_inv @ t

    For camera poses (4x4), rotation is transformed without scale while
    position gets the full Sim(3) treatment.
    """

    def __init__(self, alignment_dir: str):
        alignment_path = Path(alignment_dir)
        sim3_path = alignment_path / "colmap_to_mocap_sim3.json"

        if not sim3_path.exists():
            raise FileNotFoundError(f"Sim(3) file not found: {sim3_path}")

        with open(sim3_path) as f:
            sim3 = json.load(f)

        self.s: float = sim3["scale"]
        self.R: np.ndarray = np.array(sim3["R"])
        self.t: np.ndarray = np.array(sim3["t"])

        # Precompute inverse
        self.s_inv: float = 1.0 / self.s
        self.R_inv: np.ndarray = self.R.T
        self.t_inv: np.ndarray = -self.s_inv * (self.R_inv @ self.t)

        logger.info(
            "Loaded coordinate transformation from %s (scale COLMAP->MOCAP: %.4f)",
            alignment_path.name, self.s,
        )

    def mocap_to_colmap_position(self, pos_mocap: np.ndarray) -> np.ndarray:
        """Transform a 3D position from MOCAP frame to COLMAP frame."""
        pos_mocap = np.asarray(pos_mocap, dtype=float)
        return self.s_inv * (self.R_inv @ pos_mocap) + self.t_inv

    def mocap_to_colmap_pose(self, T_mocap: np.ndarray) -> np.ndarray:
        """Transform a 4x4 camera-to-world pose from MOCAP to COLMAP frame."""
        T_colmap = np.eye(4)
        T_colmap[:3, :3] = self.R_inv @ T_mocap[:3, :3]
        T_colmap[:3, 3] = self.s_inv * (self.R_inv @ T_mocap[:3, 3]) + self.t_inv
        return T_colmap

    def colmap_to_mocap_pose(self, T_colmap: np.ndarray) -> np.ndarray:
        """Transform a 4x4 camera-to-world pose from COLMAP to MOCAP frame."""
        T_mocap = np.eye(4)
        T_mocap[:3, :3] = self.R @ T_colmap[:3, :3]
        T_mocap[:3, 3] = self.s * (self.R @ T_colmap[:3, 3]) + self.t
        return T_mocap

    def get_transformation_info(self) -> dict:
        """Get transformation parameters for debugging."""
        return {
            "colmap_to_mocap": {
                "scale": self.s,
                "rotation": self.R,
                "translation": self.t,
            },
            "mocap_to_colmap": {
                "scale": self.s_inv,
                "rotation": self.R_inv,
                "translation": self.t_inv,
            },
        }


# ===================================================================
# Scene factory
# ===================================================================

# Alignment directories are resolved relative to the repo root.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent

_SCENE_ALIGNMENT_DIRS: Dict[str, List[Path]] = {
    "left_gate": [_REPO_ROOT / "data" / "alignment" / "left_gate"],
    "right_gate": [_REPO_ROOT / "data" / "alignment" / "right_gate"],
}


def register_scene_alignment(scene_name: str, alignment_dir: str | Path) -> None:
    """Register an additional alignment directory for a scene.

    This allows users to extend the scene registry without modifying library code.
    """
    path = Path(alignment_dir)
    if scene_name not in _SCENE_ALIGNMENT_DIRS:
        _SCENE_ALIGNMENT_DIRS[scene_name] = []
    _SCENE_ALIGNMENT_DIRS[scene_name].insert(0, path)


def create_transformer_for_scene(scene_name: str) -> CoordinateTransformer:
    """Create a CoordinateTransformer for a named scene.

    Searches registered alignment directories for the scene's
    ``colmap_to_mocap_sim3.json`` file.
    """
    if scene_name not in _SCENE_ALIGNMENT_DIRS:
        raise ValueError(
            f"Unknown scene: {scene_name}. "
            f"Valid options: {list(_SCENE_ALIGNMENT_DIRS.keys())}. "
            f"Use register_scene_alignment() to add custom scenes."
        )

    for candidate in _SCENE_ALIGNMENT_DIRS[scene_name]:
        if (candidate / "colmap_to_mocap_sim3.json").exists():
            return CoordinateTransformer(str(candidate))

    tried = [str(p) for p in _SCENE_ALIGNMENT_DIRS[scene_name]]
    raise FileNotFoundError(
        f"Alignment data not found for '{scene_name}'. Searched: {tried}"
    )


# ===================================================================
# NED ↔ Z-up conversions
# ===================================================================

# Permutation matrices for FiGS NED ↔ MOCAP Z-up.
# The permutation ID encodes sign flips on the y and z axes.
_PERM_MATRICES = {
    0: np.diag([1.0, 1.0, -1.0]),
    2: np.diag([-1.0, -1.0, -1.0]),
    5: np.diag([1.0, -1.0, -1.0]),
}


def _get_perm_diag(perm: int) -> np.ndarray:
    """Return the 3x3 diagonal sign matrix for a permutation ID."""
    return _PERM_MATRICES.get(perm, np.diag([1.0, 1.0, -1.0]))


def convert_zup_to_ned(pos_zup: np.ndarray, perm: int = 5) -> np.ndarray:
    """Convert a Z-up (MOCAP) position to FiGS NED."""
    p = np.asarray(pos_zup, dtype=float)
    return _get_perm_diag(perm) @ p


def convert_ned_to_zup(pos_ned: np.ndarray, perm: int = 5) -> np.ndarray:
    """Convert a FiGS NED position to Z-up (MOCAP). Inverse of convert_zup_to_ned."""
    p = np.asarray(pos_ned, dtype=float)
    # The diagonal matrix is its own inverse
    return _get_perm_diag(perm) @ p


# ===================================================================
# FiGS → Nerfstudio-internal transform
# ===================================================================


def build_figs_to_nerf_transform(
    scene_key: str,
    permutation: int,
    config_yml_path: str | Path | None = None,
) -> np.ndarray:
    """Compose FiGS NED -> MOCAP Z-up -> COLMAP -> Nerfstudio-internal transform.

    The nerfstudio training pipeline applies an additional dataparser transform
    (stored in ``dataparser_transforms.json`` next to ``config.yml``).
    When ``config_yml_path`` is provided, that extra step is applied.

    Parameters
    ----------
    scene_key : str
        Scene name for loading the COLMAP↔MOCAP Sim(3) alignment.
    permutation : int
        Permutation ID for FiGS NED ↔ MOCAP sign convention.
    config_yml_path : path, optional
        Path to the nerfstudio config.yml. If provided, the dataparser
        transform is loaded and applied.

    Returns
    -------
    T : (4, 4) array
        Homogeneous transform from FiGS NED to nerfstudio-internal frame.
    """
    transformer = create_transformer_for_scene(scene_key)

    # FiGS NED -> MOCAP Z-up
    t_figs_to_mocap = np.eye(4)
    t_figs_to_mocap[:3, :3] = _get_perm_diag(permutation)

    # MOCAP Z-up -> COLMAP (Sim(3) inverse)
    t_mocap_to_colmap = np.eye(4)
    t_mocap_to_colmap[:3, :3] = transformer.s_inv * transformer.R_inv
    t_mocap_to_colmap[:3, 3] = transformer.t_inv

    T = t_mocap_to_colmap @ t_figs_to_mocap

    # COLMAP -> Nerfstudio-internal (dataparser_transforms.json)
    if config_yml_path is not None:
        dp_path = Path(config_yml_path).parent / "dataparser_transforms.json"
        if dp_path.exists():
            dp = json.loads(dp_path.read_text())
            dp_mat = np.array(dp["transform"])  # (3, 4)
            dp_scale = float(dp["scale"])
            t_colmap_to_ns = np.eye(4)
            t_colmap_to_ns[:3, :3] = dp_scale * dp_mat[:, :3]
            t_colmap_to_ns[:3, 3] = dp_scale * dp_mat[:, 3]
            T = t_colmap_to_ns @ T

    return T


# ===================================================================
# Camera transform utilities
# ===================================================================


def build_camera_transforms(
    forward_offset: float = -0.05,
    downward_offset: float = -0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the dual-camera camera-to-body transforms for the standard drone setup.

    Camera convention (OpenGL/nerfstudio): +x right, +y up, +z backward.
    Body convention (NED): +x forward, +y right, +z down.

    Forward camera: looks along body +x (forward).
    Downward camera: forward camera rotated -90 deg about camera x (looks down).

    Returns
    -------
    Tc2b_forward : (4, 4) array
    Tc2b_downward : (4, 4) array
    """
    # Forward camera: cam_x->body_y, cam_y->body_-z, cam_z->body_-x
    Tc2b_base = np.eye(4)
    Tc2b_base[:3, :3] = np.array([
        [0, 0, -1],
        [1, 0,  0],
        [0, -1, 0],
    ])
    Tc2b_base[2, 3] = forward_offset

    # +90 deg about camera z to correct orientation
    Rz_pos90 = np.eye(4)
    Rz_pos90[:3, :3] = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ])
    Tc2b_forward = Tc2b_base @ Rz_pos90

    # Downward camera: forward camera rotated -90 deg about camera x
    Rx_neg90 = np.eye(4)
    Rx_neg90[:3, :3] = np.array([
        [1,  0, 0],
        [0,  0, 1],
        [0, -1, 0],
    ])
    Tc2b_downward = Tc2b_forward @ Rx_neg90

    return Tc2b_forward, Tc2b_downward
