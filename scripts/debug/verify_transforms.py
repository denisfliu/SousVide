#!/usr/bin/env python3
"""Trace gate/goal/start positions through every coordinate frame.

Prints positions in: MOCAP Z-up, FiGS NED, COLMAP, NS-internal.
Also does round-trip checks and compares with the viewer's to_viewer chain.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vla_falsification.falsification.config import (
    GATE_PRESETS, load_config, apply_gate_preset, convert_to_ned, convert_from_ned_to_zup,
)
from vla_falsification.utilities.coordinate_transform import (
    CoordinateTransformer, create_transformer_for_scene,
    build_figs_to_nerf_transform, _get_perm_diag,
    convert_zup_to_ned, convert_ned_to_zup,
)

GATE = "left_gate"

cfg = load_config(None)
apply_gate_preset(cfg, GATE)
preset = GATE_PRESETS[GATE]
perm = cfg["simulation"]["permutation"]
scene_key = preset["scene_key"]
gsplat_path = Path(cfg["scene"]["config_yml"])

P = _get_perm_diag(perm)
transformer = create_transformer_for_scene(scene_key)

# Load dataparser transform for the viewer chain
dp_path = gsplat_path.parent / "dataparser_transforms.json"
dp = json.loads(dp_path.read_text())
dp_mat = np.array(dp["transform"])  # (3, 4)
dp_scale = float(dp["scale"])

# Build the full FiGS-to-NS transform
T_figs_to_ns = build_figs_to_nerf_transform(scene_key, perm, gsplat_path)
T_ns_to_figs = np.linalg.inv(T_figs_to_ns)

axis_flip = np.diag([1.0, -1.0, -1.0])

points = {
    "start": np.array(preset["start_position_zup"]),
    "gate":  np.array(preset["gate_position_zup"]),
    "goal":  np.array(preset["goal_position_zup"]),
}

print("=" * 70)
print(f"COORDINATE TRANSFORM VERIFICATION — {GATE}")
print(f"Permutation: {perm}  →  P = diag{tuple(np.diag(P))}")
print(f"Sim3 scale (COLMAP→MOCAP): {transformer.s:.6f}")
print(f"Dataparser scale: {dp_scale:.6f}")
print("=" * 70)

for name, pos_zup in points.items():
    print(f"\n--- {name.upper()} ---")

    # Frame 1: MOCAP Z-up (the "ground truth" from GATE_PRESETS)
    print(f"  MOCAP Z-up:     {pos_zup}")

    # Frame 2: FiGS NED
    pos_ned = convert_zup_to_ned(pos_zup, perm)
    print(f"  FiGS NED:       {pos_ned}")

    # Round-trip NED → MOCAP
    pos_zup_rt = convert_ned_to_zup(pos_ned, perm)
    rt_err = np.linalg.norm(pos_zup - pos_zup_rt)
    print(f"  NED→MOCAP RT:   {pos_zup_rt}  (err={rt_err:.2e})")

    # Frame 3: COLMAP (from MOCAP via Sim3 inverse)
    pos_colmap = transformer.mocap_to_colmap_position(pos_zup)
    print(f"  COLMAP:         {pos_colmap}")

    # Round-trip COLMAP → MOCAP
    pos_zup_rt2 = transformer.s * (transformer.R @ pos_colmap) + transformer.t
    rt_err2 = np.linalg.norm(pos_zup - pos_zup_rt2)
    print(f"  COLMAP→MOCAP RT:{pos_zup_rt2}  (err={rt_err2:.2e})")

    # Frame 4a: Viewer chain (MOCAP → COLMAP → axis_flip → dataparser)
    pos_flipped = axis_flip @ pos_colmap
    pos_viewer = dp_scale * (dp_mat[:, :3] @ pos_flipped + dp_mat[:, 3])
    print(f"  NS (viewer):    {pos_viewer}")

    # Frame 4b: build_figs_to_nerf_transform chain (NED → NS)
    pos_ns = (T_figs_to_ns @ np.append(pos_ned, 1.0))[:3]
    print(f"  NS (figs→ns):   {pos_ns}")

    # Difference between the two chains
    diff = pos_viewer - pos_ns
    print(f"  Chain diff:     {diff}  (|diff|={np.linalg.norm(diff):.2e})")

    # Round-trip NS → FiGS
    pos_ned_rt = (T_ns_to_figs @ np.append(pos_ns, 1.0))[:3]
    rt_err3 = np.linalg.norm(pos_ned - pos_ned_rt)
    print(f"  NS→NED RT:      {pos_ned_rt}  (err={rt_err3:.2e})")

# Distance checks
print("\n" + "=" * 70)
print("DISTANCE CHECKS")
print("=" * 70)

for frame_name, get_pos in [
    ("MOCAP Z-up", lambda n: points[n]),
    ("FiGS NED",   lambda n: convert_zup_to_ned(points[n], perm)),
    ("NS (figs)",  lambda n: (T_figs_to_ns @ np.append(convert_zup_to_ned(points[n], perm), 1.0))[:3]),
]:
    s = get_pos("start")
    g = get_pos("gate")
    gl = get_pos("goal")
    print(f"\n  {frame_name}:")
    print(f"    start→gate:  {np.linalg.norm(g - s):.6f}")
    print(f"    gate→goal:   {np.linalg.norm(gl - g):.6f}")
    print(f"    start→goal:  {np.linalg.norm(gl - s):.6f}")

# Check: what position does the sim actually use?
print("\n" + "=" * 70)
print("SIM CONFIG CHECK (what run_falsification.py computes)")
print("=" * 70)
gate_zup = cfg["simulation"]["gate_position_zup"]
gate_ned_cfg = convert_to_ned(gate_zup, perm)
print(f"  gate_position_zup (config): {gate_zup}")
print(f"  gate NED (for sim):         {gate_ned_cfg}")
print(f"  gate NED (from preset):     {convert_zup_to_ned(np.array(preset['gate_position_zup']), perm)}")

# x0 in the sim
x0_ned = cfg["simulation"].get("start_position_zup")
if x0_ned:
    x0_ned_conv = convert_to_ned(x0_ned, perm)
    print(f"  start_position_zup:         {x0_ned}")
    print(f"  x0 NED (for sim):           {x0_ned_conv}")

# Check the actual trajectory start/end vs expected
traj_path = Path("falsification_results/left_gate_baseline/episodes/episode_00000/trajectory.npz")
if traj_path.exists():
    traj = np.load(traj_path)
    states = traj["states"]
    pos_mocap = traj["positions_mocap"]
    print(f"\n  Actual traj start NED:      {states[0, :3]}")
    print(f"  Actual traj start MOCAP:    {pos_mocap[0]}")
    print(f"  Actual traj end NED:        {states[-1, :3]}")
    print(f"  Actual traj end MOCAP:      {pos_mocap[-1]}")

    # Distance from traj end to gate
    gate_mocap = np.array(preset["gate_position_zup"])
    d_gate = np.linalg.norm(pos_mocap[-1] - gate_mocap)
    d_goal = np.linalg.norm(pos_mocap[-1] - np.array(preset["goal_position_zup"]))
    print(f"  End→gate distance (MOCAP):  {d_gate:.4f} m")
    print(f"  End→goal distance (MOCAP):  {d_goal:.4f} m")

    # Closest approach to gate
    dists_to_gate = np.linalg.norm(pos_mocap - gate_mocap, axis=1)
    min_idx = np.argmin(dists_to_gate)
    print(f"  Closest to gate: step {min_idx}, dist={dists_to_gate[min_idx]:.4f} m, pos={pos_mocap[min_idx]}")

# What does the viewer see? Compute gate in NS frame via both chains
print("\n" + "=" * 70)
print("VIEWER MARKER POSITIONS (what should appear in Nerfstudio)")
print("=" * 70)
for name, pos_zup in points.items():
    pos_colmap = transformer.mocap_to_colmap_position(pos_zup)
    pos_flipped = axis_flip @ pos_colmap
    pos_ns = dp_scale * (dp_mat[:, :3] @ pos_flipped + dp_mat[:, 3])
    print(f"  {name:8s} NS: {pos_ns}")
