# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VLA Falsification** is a research framework for systematic falsification and recovery of Vision-Language-Action (VLA) control policies for drones in photorealistic Gaussian Splatting (GSplat) environments. Built on top of the FiGS flight simulator.

## Setup & Environment

```bash
# Prerequisites: CUDA 11.8 toolkit, COLMAP, acados (built locally in external/FiGS/acados/)
git submodule update --recursive --init
uv sync                    # installs all deps including PyTorch CUDA 12.1, tiny-cuda-nn from source
source .venv/bin/activate
```

Required environment variables (typically in ~/.zshrc):
- `LD_LIBRARY_PATH` must include `external/FiGS/acados/lib`
- `ACADOS_SOURCE_DIR` pointing to `external/FiGS/acados/`
- `CUDA_HOME` pointing to local CUDA 11.8 install

Run tests with:
```bash
PYTHONPATH="src:.:$PYTHONPATH" python -m pytest tests/ -v
```

Tests cover: perturbations, failure detection, coordinate transforms, NED conversions, VLA helpers, campaign summarization. No linter is configured.

Docker build:
```bash
docker build -f docker/Dockerfile -t vla-falsification .
docker run --gpus all vla-falsification --gate left_gate --num-episodes 10
```

## Architecture

### Falsification System (`src/vla_falsification/falsification/`)

The core research contribution. Systematically stress-tests VLA policies:

- **`config.py`**: `GATE_PRESETS`, `DEFAULT_CONFIG`, `apply_gate_preset()`, `load_config()` — shared configuration for all falsification scripts
- **`orchestrator.py`**: `FalsificationOrchestrator` runs episodes: reset -> apply perturbations -> run VLA -> detect failure -> plan recovery
- **`perturbations.py`**: Three perturbation surfaces — **Action** (noise/bias/scale on thrust+rates), **Observation** (image/state/camera noise), **Environment** (shift/scale/opacity of Gaussians via Splat-MOVER)
- **`failure_detector.py`**: Pluggable `SafetyCriterion` voting — bounds, velocity, tilt, proximity collision
- **`splatnav_recovery.py`**: Plans collision-free recovery trajectories from failure states using SplatNav

### Control (`src/vla_falsification/control/`)

- **`vla_policy.py`**: `VLAPolicy` wraps external VLA servers (OpenPI) via websocket

### Utilities (`src/vla_falsification/utilities/`)

- **`coordinate_transform.py`**: `CoordinateTransformer` for converting between FiGS/NED, MOCAP/Z-up, COLMAP, and Nerfstudio frames. Alignment data loaded from JSON files in `data/alignment/`. Getting coordinate chains right is critical — failures here propagate silently.

### Git Submodules

- **external/FiGS/**: Flight-in-Gaussian-Splats simulator — ACADOS dynamics, GSplat rendering, MPC trajectory tracking, MinTimeSnap trajectory generation
- **external/splatnav/**: Collision-aware A* + spline path planning over Gaussian splat scenes
- **external/Splat-MOVER/**: Gaussian splat scene editing (used for environment perturbations)

## Key Entry Points

| Script | Purpose |
|--------|---------|
| `run_falsification.py` | Main falsification campaign — configures FiGS, VLA, perturbations, safety criteria, runs N episodes |
| `view_falsification_trajectories.py` | Visualize falsification results in Nerfstudio viewer |
| `scripts/recover_falsification_episode.py` | Post-hoc recovery planning on saved episodes |
| `scripts/render_falsification_cameras.py` | Render camera views from saved episodes |
| `scripts/visualization/ns_viewer_with_trajectories.py` | Visualize trajectories in Nerfstudio viewer |

## Configuration

- **`src/vla_falsification/falsification/config.py`**: Gate presets and default falsification config
- **`configs/falsification/`**: YAML overrides for falsification runs (config_gate_only.yaml, etc.)
- **`configs/frames/`**: Drone specifications (mass, thrust coefficient, rotors) — used by `vla_policy.py` and `render_falsification_cameras.py` via `figs.utilities.config_helper.get_config()`

All FiGS configs loaded via `figs.utilities.config_helper.get_config()`.

## Data Layout

```
data/alignment/left_gate/              # COLMAP-to-MOCAP Sim(3) transforms
data/alignment/right_gate/
data/colmap/left_gate/                 # COLMAP reconstructions & trained GSplats
data/colmap/right_gate/
falsification_results/<gate>/episodes/ # Falsification output
```
