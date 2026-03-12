#!/usr/bin/env python3
"""Compare distances in MOCAP/FiGS frame vs Nerfstudio frame."""

import numpy as np
from pathlib import Path

from vla_falsification.falsification.config import GATE_PRESETS, load_config, apply_gate_preset, convert_to_ned
from vla_falsification.utilities.coordinate_transform import build_figs_to_nerf_transform

cfg = load_config(None)
apply_gate_preset(cfg, 'left_gate')
preset = GATE_PRESETS['left_gate']
perm = cfg['simulation']['permutation']
gsplat_path = Path(cfg['scene']['config_yml'])
T = build_figs_to_nerf_transform(preset['scene_key'], perm, gsplat_path)
T_inv = np.linalg.inv(T)

def figs_to_nerf(pos):
    p_h = np.append(np.asarray(pos, dtype=float), 1.0)
    return (T @ p_h)[:3]

def nerf_to_figs(pos):
    p_h = np.append(np.asarray(pos, dtype=float), 1.0)
    return (T_inv @ p_h)[:3]

gate_ned = convert_to_ned(cfg['simulation']['gate_position_zup'], perm)
start_ned = convert_to_ned(cfg['simulation']['start_position_zup'], perm)
goal_ned = convert_to_ned(cfg['simulation']['goal_position_zup'], perm)

gate_nerf = figs_to_nerf(gate_ned)
start_nerf = figs_to_nerf(start_ned)
goal_nerf = figs_to_nerf(goal_ned)

print("=== Positions ===")
print(f"Start FiGS: {start_ned}")
print(f"Start NS:   {start_nerf}")
print(f"Gate  FiGS: {gate_ned}")
print(f"Gate  NS:   {gate_nerf}")
print(f"Goal  FiGS: {goal_ned}")
print(f"Goal  NS:   {goal_nerf}")

print("\n=== Distances (FiGS NED = MOCAP meters) ===")
d_start_gate_figs = np.linalg.norm(gate_ned - start_ned)
d_gate_goal_figs = np.linalg.norm(goal_ned - gate_ned)
d_start_goal_figs = np.linalg.norm(goal_ned - start_ned)
print(f"start → gate:  {d_start_gate_figs:.4f} m")
print(f"gate  → goal:  {d_gate_goal_figs:.4f} m")
print(f"start → goal:  {d_start_goal_figs:.4f} m")

print("\n=== Distances (Nerfstudio frame) ===")
d_start_gate_nerf = np.linalg.norm(gate_nerf - start_nerf)
d_gate_goal_nerf = np.linalg.norm(goal_nerf - gate_nerf)
d_start_goal_nerf = np.linalg.norm(goal_nerf - start_nerf)
print(f"start → gate:  {d_start_gate_nerf:.6f} NS")
print(f"gate  → goal:  {d_gate_goal_nerf:.6f} NS")
print(f"start → goal:  {d_start_goal_nerf:.6f} NS")

print("\n=== Scale factor (FiGS / NS) ===")
print(f"start → gate:  {d_start_gate_figs / d_start_gate_nerf:.2f} m per NS unit")
print(f"gate  → goal:  {d_gate_goal_figs / d_gate_goal_nerf:.2f} m per NS unit")
print(f"start → goal:  {d_start_goal_figs / d_start_goal_nerf:.2f} m per NS unit")

print("\n=== Transform matrix ===")
print(T)
U, S, Vt = np.linalg.svd(T[:3, :3])
print(f"\nSingular values of rotation+scale block: {S}")
print(f"Scale factor (det^(1/3)): {np.linalg.det(T[:3,:3])**(1/3):.6f}")

# Check: validation uses FiGS positions but compares with meter thresholds
# Are the trajectory positions from plan_via_gate actually in FiGS meters?
print("\n=== Round-trip check ===")
gate_rt = nerf_to_figs(gate_nerf)
print(f"Gate FiGS original:    {gate_ned}")
print(f"Gate FiGS round-trip:  {gate_rt}")
print(f"Round-trip error:      {np.linalg.norm(gate_ned - gate_rt):.2e}")
