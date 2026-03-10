"""Falsification pipeline configuration.

Contains gate presets, default configuration, and helpers for building
the final config dict from CLI arguments and YAML overrides.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import yaml

from sousvide.utilities.coordinate_transform import (
    convert_ned_to_zup,
    convert_zup_to_ned,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Legacy aliases — re-exported so callers that used
# ``from run_falsification import convert_to_ned`` can switch to
# ``from sousvide.falsification.config import convert_to_ned``.
convert_to_ned = convert_zup_to_ned
convert_from_ned_to_zup = convert_ned_to_zup

# ===================================================================
# Per-gate scene presets
# ===================================================================

GATE_PRESETS: Dict[str, Dict] = {
    "left_gate": {
        "gsplat_name": "data/colmap/left_gate/sagesplat/2025-10-06_215922",
        "config_yml": _REPO_ROOT / "data" / "colmap" / "left_gate" / "sagesplat" / "2025-10-06_215922" / "config.yml",
        "scene_key": "left_gate",
        "permutation": 5,
        "start_position_zup": [0.104, -0.0219, 1.364],
        "goal_position_zup": [1.421417, -0.3320115, 1.0],
        "gate_position_zup": [0.936, 0.015, 1.134],
        "gate_opening_half_w": 0.25,
        "gate_opening_half_h": 0.25,
        "gate_post_radius": 0.04,
        "gate_mask_path": str(_REPO_ROOT / "artifacts" / "left_gate" / "left_gate_bottom_mask.npy"),
        "gate_points_path": str(_REPO_ROOT / "artifacts" / "left_gate" / "left_gate_bottom_points.npy"),
        "table_points_path": str(_REPO_ROOT / "artifacts" / "left_gate" / "left_table_points.npy"),
        "prompt": "go through the gate on the left and hover over the stuffed animal",
    },
    "right_gate": {
        "gsplat_name": "data/colmap/right_gate/sagesplat/2025-10-01_103533",
        "config_yml": _REPO_ROOT / "data" / "colmap" / "right_gate" / "sagesplat" / "2025-10-01_103533" / "config.yml",
        "scene_key": "right_gate",
        "permutation": 5,
        "start_position_zup": [0.104, -0.0219, 1.364],
        "goal_position_zup": [1.421417, -0.3320115, 1.0],
        "gate_position_zup": [1.107, -0.343, 1.159],
        "gate_opening_half_w": 0.25,
        "gate_opening_half_h": 0.25,
        "gate_post_radius": 0.04,
        "gate_mask_path": str(_REPO_ROOT / "artifacts" / "right_gate" / "right_gate_bottom_mask.npy"),
        "gate_points_path": str(_REPO_ROOT / "artifacts" / "right_gate" / "right_gate_bottom_points.npy"),
        "table_points_path": str(_REPO_ROOT / "artifacts" / "right_gate" / "right_table_points.npy"),
        "prompt": "go through the gate on the right and hover over the stuffed animal",
    },
}


# ===================================================================
# Default configuration
# ===================================================================

DEFAULT_CONFIG: Dict = {
    "scene": {
        "gsplat_name": None,
        "config_yml": None,
        "gsplats_path": str(_REPO_ROOT),
        "scene_key": None,
        "splatnav_gsplat_path": None,
    },
    "simulation": {
        "t0": 0.0,
        "tf": 12.0,
        "frame_name": "carl",
        "permutation": 5,
        "goal_position_zup": None,
        "start_position_zup": None,
        "gate_position_zup": None,
        "gate_pass_radius_m": 0.25,
    },
    "vla": {
        "host": "moraband",
        "port": 8000,
        "prompt": None,
        "hz": 10,
        "actions_per_chunk": 50,
        "action_mapper_type": "position_delta",
        "action_mapper_kwargs": {},
        "image_size": 256,
        "mask_third_person": True,
    },
    "safety": {
        "bounds_lower": [-5.0, -5.0, -5.0],
        "bounds_upper": [5.0, 5.0, 5.0],
        "max_speed": 5.0,
        "max_tilt_deg": 60.0,
        "safe_horizon": 3,
        "robot_radius": 0.15,
    },
    "perturbations": {
        "action": [
            {"type": "ActionNoise", "std": [0.05, 0.1, 0.1, 0.1]},
        ],
        "observation_image": [
            {"type": "ImageNoise", "std": 10.0},
        ],
        "observation_state": [],
        "observation_camera": [],
        "environment_means": [],
        "environment_scales": [],
        "environment_opacities": [],
    },
    "recovery": {
        "enable": True,
        "robot_radius": 0.02,
        "vmax": 2.0,
        "amax": 3.0,
        "recovery_total_time": 5.0,
        "env_lower_bound": [-0.5, -0.5, -0.5],
        "env_upper_bound": [0.5, 0.5, 0.5],
        "voxel_resolution": 150,
    },
    "campaign": {
        "num_episodes": 50,
        "seed_offset": 0,
    },
    "output": {
        "dir": None,
    },
}


# ===================================================================
# Helpers
# ===================================================================

def apply_gate_preset(cfg: Dict, gate: str) -> Dict:
    """Overlay a gate preset onto the config, without clobbering user overrides."""
    preset = GATE_PRESETS[gate]
    cfg["scene"]["gsplat_name"] = cfg["scene"]["gsplat_name"] or preset["gsplat_name"]
    cfg["scene"]["config_yml"] = str(preset["config_yml"])
    cfg["scene"]["scene_key"] = cfg["scene"]["scene_key"] or preset["scene_key"]
    cfg["scene"]["splatnav_gsplat_path"] = (
        cfg["scene"]["splatnav_gsplat_path"] or str(preset["config_yml"])
    )
    cfg["simulation"]["permutation"] = preset["permutation"]
    cfg["simulation"]["start_position_zup"] = (
        cfg["simulation"]["start_position_zup"] or preset["start_position_zup"]
    )
    cfg["simulation"]["goal_position_zup"] = (
        cfg["simulation"]["goal_position_zup"] or preset["goal_position_zup"]
    )
    cfg["simulation"]["gate_position_zup"] = (
        cfg["simulation"]["gate_position_zup"] or preset["gate_position_zup"]
    )
    cfg["vla"]["prompt"] = cfg["vla"]["prompt"] or preset["prompt"]
    gate_mask_path = Path(preset["gate_mask_path"])
    gate_points_path = Path(preset["gate_points_path"])
    table_points_path = Path(preset["table_points_path"])

    if (
        gate in ("left_gate", "right_gate")
        and not cfg["perturbations"].get("environment_means")
        and gate_mask_path.exists()
        and gate_points_path.exists()
        and table_points_path.exists()
    ):
        cfg["perturbations"]["environment_means"] = [
            {
                "type": "GateRigidTransform",
                "gate_mask_path": str(gate_mask_path),
                "gate_points_path": str(gate_points_path),
                "table_points_path": str(table_points_path),
                "max_match_distance_m": 0.01,
                "max_translation_m": [0.04, 0.04, 0.02],
                "yaw_range_deg": [-6.0, 6.0],
                "min_translation_m": 0.002,
                "min_abs_yaw_deg": 0.5,
                "min_table_clearance_m": 0.03,
                "max_sampling_tries": 120,
                "strict": True,
            }
        ]
    cfg["output"]["dir"] = cfg["output"]["dir"] or f"falsification_results/{gate}"
    return cfg


def load_config(path: str | None) -> Dict:
    """Deep-copy defaults, then merge a user YAML config on top.

    If *path* is a filename without directory separators and does not exist
    as-is, the function also checks ``configs/falsification/`` for a match.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            alt = _REPO_ROOT / "configs" / "falsification" / config_path.name
            if alt.exists():
                config_path = alt
        with open(config_path) as f:
            user = yaml.safe_load(f) or {}
        for section, vals in user.items():
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(vals)
            else:
                cfg[section] = vals
    return cfg
