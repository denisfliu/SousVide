#!/usr/bin/env python3
"""Quick test of SplatNav recovery with DEBUG logging."""

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s:%(levelname)s: %(message)s')
logging.getLogger('vla_falsification.falsification.splatnav_recovery').setLevel(logging.DEBUG)

import numpy as np
from pathlib import Path

from vla_falsification.falsification.config import GATE_PRESETS, load_config, apply_gate_preset, convert_to_ned
from vla_falsification.utilities.coordinate_transform import build_figs_to_nerf_transform
from vla_falsification.falsification.splatnav_recovery import SplatNavRecovery, RecoveryConfig

cfg = load_config(None)
apply_gate_preset(cfg, 'left_gate')
preset = GATE_PRESETS['left_gate']
perm = cfg['simulation']['permutation']
gsplat_path = Path(cfg['scene']['config_yml'])
T = build_figs_to_nerf_transform(preset['scene_key'], perm, gsplat_path)

gate_ned = convert_to_ned(cfg['simulation']['gate_position_zup'], perm)
start_ned = convert_to_ned(cfg['simulation']['start_position_zup'], perm)
goal_ned = convert_to_ned(cfg['simulation']['goal_position_zup'], perm)

print(f"Start: {start_ned}")
print(f"Gate:  {gate_ned}")
print(f"Goal:  {goal_ned}")

rec = SplatNavRecovery(gsplat_path, RecoveryConfig(voxel_resolution=100), coordinate_transform=T)
result = rec.plan_via_gate(start_ned, gate_ned, goal_ned)
print(f"\nFeasible: {result.feasible}")
print(f"Metadata: {result.metadata}")
