#!/usr/bin/env python3
"""Launch nerfstudio viewer with an in-memory gate perturbation applied."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from threading import Lock

import numpy as np

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CC", "gcc-11")
os.environ.setdefault("CXX", "g++-11")

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState

from vla_falsification.falsification.perturbations import (
    GateRigidTransform,
    GateRigidTransformConfig,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATHS = {
    "left_gate": REPO_ROOT
    / "data" / "colmap" / "left_gate"
    / "sagesplat"
    / "2025-10-06_215922"
    / "config.yml",
    "right_gate": REPO_ROOT
    / "data" / "colmap" / "right_gate"
    / "sagesplat"
    / "2025-10-01_103533"
    / "config.yml",
}
ARTIFACT_PATHS = {
    "left_gate": {
        "gate_mask": REPO_ROOT / "artifacts" / "left_gate" / "left_gate_bottom_mask.npy",
        "gate_points": REPO_ROOT / "artifacts" / "left_gate" / "left_gate_bottom_points.npy",
        "table_points": REPO_ROOT / "artifacts" / "left_gate" / "left_table_points.npy",
    },
    "right_gate": {
        "gate_mask": REPO_ROOT / "artifacts" / "right_gate" / "right_gate_bottom_mask.npy",
        "gate_points": REPO_ROOT / "artifacts" / "right_gate" / "right_gate_bottom_points.npy",
        "table_points": REPO_ROOT / "artifacts" / "right_gate" / "right_table_points.npy",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=["left_gate", "right_gate"], required=True)
    parser.add_argument(
        "--load-config",
        type=Path,
        default=None,
        help="Optional override path to nerfstudio config.yml.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for transform sample.")
    parser.add_argument("--port", type=int, default=7010, help="Viewer websocket port.")
    parser.add_argument("--dry-run", action="store_true", help="Apply perturbation, print stats, then exit.")
    parser.add_argument("--max-tx", type=float, default=0.04)
    parser.add_argument("--max-ty", type=float, default=0.04)
    parser.add_argument("--max-tz", type=float, default=0.02)
    parser.add_argument("--yaw-min-deg", type=float, default=-6.0)
    parser.add_argument("--yaw-max-deg", type=float, default=6.0)
    parser.add_argument("--min-translation", type=float, default=0.002)
    parser.add_argument("--min-abs-yaw-deg", type=float, default=0.5)
    parser.add_argument("--min-table-clearance", type=float, default=0.03)
    args = parser.parse_args()

    config_path = args.load_config if args.load_config is not None else CONFIG_PATHS[args.gate]
    paths = ARTIFACT_PATHS[args.gate]
    if not config_path.exists():
        raise FileNotFoundError(f"Scene config missing: {config_path}")
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{key} file missing: {p}")

    config, pipeline, _, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode="test",
    )
    model = pipeline.model

    means_before = model.means.data.clone()
    quats_before = model.quats.data.clone()
    gate_mask = np.load(paths["gate_mask"]).astype(bool).reshape(-1)

    perturb = GateRigidTransform(
        GateRigidTransformConfig(
            gate_mask_path=str(paths["gate_mask"]),
            gate_points_path=str(paths["gate_points"]),
            table_points_path=str(paths["table_points"]),
            max_match_distance_m=0.01,
            max_translation_m=(args.max_tx, args.max_ty, args.max_tz),
            yaw_range_deg=(args.yaw_min_deg, args.yaw_max_deg),
            min_translation_m=args.min_translation,
            min_abs_yaw_deg=args.min_abs_yaw_deg,
            min_table_clearance_m=args.min_table_clearance,
            max_sampling_tries=120,
            strict=True,
        )
    )
    perturb.reset(np.random.RandomState(args.seed))

    means_after = perturb.apply(means_before)
    quats_after = perturb.apply_quats(quats_before)
    model.means.data = means_after
    model.quats.data = quats_after

    gate_idx = perturb._gate_indices
    if gate_idx is None:
        raise RuntimeError("Gate indices were not resolved by GateRigidTransform.")
    displacement = (means_after[gate_idx] - means_before[gate_idx]).norm(dim=-1)
    print(
        f"Applied perturbation ({args.gate}, seed={args.seed}): "
        f"translation={perturb._translation}, yaw_deg={np.degrees(perturb._yaw_rad):.3f}"
    )
    if gate_mask.shape[0] != means_before.shape[0]:
        print(
            "Mask/model size mismatch detected; used coordinate-based gate-point "
            f"matching ({len(gate_idx)} gaussians selected)."
        )
    print(
        f"Gate displacement stats [m]: min={displacement.min().item():.4f}, "
        f"mean={displacement.mean().item():.4f}, max={displacement.max().item():.4f}"
    )

    if args.dry_run:
        return

    config.vis = "viewer"
    config.viewer.websocket_port = args.port
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
    print(viewer_state.viewer_info[0])

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
    print("Viewer running with perturbed gate. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting viewer.")


if __name__ == "__main__":
    main()
