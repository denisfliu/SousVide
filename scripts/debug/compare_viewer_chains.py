#!/usr/bin/env python3
"""Compare the two coordinate transform chains used in the codebase.

Chain A (view_falsification_trajectories.py 'to_viewer'):
    MOCAP -> COLMAP (Sim3 inv) -> axis_flip diag(1,-1,-1) -> dataparser -> NS viewer

Chain B (build_figs_to_nerf_transform):
    NED -> MOCAP (perm diag) -> COLMAP (Sim3 inv) -> dataparser -> NS internal

Are they producing the same NS-frame positions for the same physical point?
"""

import json
import numpy as np
from pathlib import Path

from vla_falsification.falsification.config import GATE_PRESETS, load_config, apply_gate_preset, convert_to_ned
from vla_falsification.utilities.coordinate_transform import (
    build_figs_to_nerf_transform, create_transformer_for_scene,
    convert_ned_to_zup, _get_perm_diag,
)

cfg = load_config(None)
apply_gate_preset(cfg, 'left_gate')
preset = GATE_PRESETS['left_gate']
perm = cfg['simulation']['permutation']
scene_key = preset['scene_key']
gsplat_path = Path(cfg['scene']['config_yml'])

# ---- Chain B: build_figs_to_nerf_transform ----
T_figs_to_nerf = build_figs_to_nerf_transform(scene_key, perm, gsplat_path)

# ---- Chain A components: to_viewer ----
transformer = create_transformer_for_scene(scene_key)
axis_flip = np.diag([1.0, -1.0, -1.0])

dp_path = gsplat_path.parent / "dataparser_transforms.json"
dp = json.loads(dp_path.read_text())
dp_mat = np.array(dp["transform"])  # (3, 4)
dp_scale = float(dp["scale"])

def chain_a_to_viewer(pos_mocap):
    """MOCAP -> COLMAP -> axis flip -> dataparser"""
    p_colmap = transformer.mocap_to_colmap_position(pos_mocap)
    p_flipped = axis_flip @ p_colmap
    p_ns = dp_scale * (dp_mat[:, :3] @ p_flipped + dp_mat[:, 3])
    return p_ns

def chain_b_to_nerf(pos_ned):
    """NED -> NS via build_figs_to_nerf_transform"""
    p_h = np.append(pos_ned, 1.0)
    return (T_figs_to_nerf @ p_h)[:3]

# ---- Test with gate position ----
gate_zup = np.array(preset['gate_position_zup'])  # MOCAP Z-up
gate_ned = convert_to_ned(gate_zup, perm)          # FiGS NED

print("=== Gate position ===")
print(f"  MOCAP Z-up: {gate_zup}")
print(f"  FiGS NED:   {gate_ned}")
print()

ns_chain_a = chain_a_to_viewer(gate_zup)
ns_chain_b = chain_b_to_nerf(gate_ned)
print(f"  Chain A (viewer):     {ns_chain_a}")
print(f"  Chain B (figs_to_ns): {ns_chain_b}")
print(f"  Difference:           {ns_chain_a - ns_chain_b}")
print(f"  |diff|:               {np.linalg.norm(ns_chain_a - ns_chain_b):.2e}")
print()

# ---- Test with start position ----
start_zup = np.array(preset['start_position_zup'])
start_ned = convert_to_ned(start_zup, perm)

print("=== Start position ===")
print(f"  MOCAP Z-up: {start_zup}")
print(f"  FiGS NED:   {start_ned}")
print()

ns_chain_a_s = chain_a_to_viewer(start_zup)
ns_chain_b_s = chain_b_to_nerf(start_ned)
print(f"  Chain A (viewer):     {ns_chain_a_s}")
print(f"  Chain B (figs_to_ns): {ns_chain_b_s}")
print(f"  Difference:           {ns_chain_a_s - ns_chain_b_s}")
print(f"  |diff|:               {np.linalg.norm(ns_chain_a_s - ns_chain_b_s):.2e}")
print()

# ---- Check axis flip difference ----
print("=== Decomposing the difference ===")
print(f"  Perm diag (NED->MOCAP): {np.diag(_get_perm_diag(perm))}")
print(f"  Axis flip:              {np.diag(axis_flip)}")
print()

# Chain A without axis flip
p_colmap = transformer.mocap_to_colmap_position(gate_zup)
p_no_flip = dp_scale * (dp_mat[:, :3] @ p_colmap + dp_mat[:, 3])
p_with_flip = dp_scale * (dp_mat[:, :3] @ (axis_flip @ p_colmap) + dp_mat[:, 3])
print(f"  COLMAP pos:          {p_colmap}")
print(f"  NS (no axis flip):   {p_no_flip}")
print(f"  NS (with axis flip): {p_with_flip}")
print(f"  Chain B result:      {ns_chain_b}")
print()

# ---- Distance comparison ----
d_mocap = np.linalg.norm(gate_zup - start_zup)
d_chain_a = np.linalg.norm(ns_chain_a - ns_chain_a_s)
d_chain_b = np.linalg.norm(ns_chain_b - ns_chain_b_s)
print("=== start→gate distance ===")
print(f"  MOCAP:   {d_mocap:.6f} m")
print(f"  Chain A: {d_chain_a:.6f} NS")
print(f"  Chain B: {d_chain_b:.6f} NS")
print(f"  Ratio A: {d_mocap / d_chain_a:.2f} m/NS")
print(f"  Ratio B: {d_mocap / d_chain_b:.2f} m/NS")
