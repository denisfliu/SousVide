# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SousVide** (Scene Understanding via Synthesized Visual Inertial Data from Experts) is a research framework for systematic falsification and recovery of Vision-Language-Action (VLA) control policies for drones in photorealistic Gaussian Splatting (GSplat) environments. Built on top of the FiGS flight simulator.

## Setup & Environment

```bash
# Prerequisites: CUDA 11.8 toolkit, COLMAP, acados (built locally in FiGS/acados/)
git submodule update --recursive --init
uv sync                    # installs all deps including PyTorch CUDA 12.1, tiny-cuda-nn from source
source .venv/bin/activate
```

Required environment variables (typically in ~/.zshrc):
- `LD_LIBRARY_PATH` must include `FiGS/acados/lib`
- `ACADOS_SOURCE_DIR` pointing to `FiGS/acados/`
- `CUDA_HOME` pointing to local CUDA 11.8 install

Run tests with:
```bash
PYTHONPATH="src:.:$PYTHONPATH" python -m pytest tests/ -v
```

Tests cover: perturbations, failure detection, coordinate transforms, NED conversions, VLA helpers, campaign summarization. No linter is configured.

Docker build:
```bash
docker build -f docker/Dockerfile -t sousvide .
docker run --gpus all sousvide --gate left_gate --num-episodes 10
```

## Architecture

### Core Pipeline: Synthesize -> Instruct -> Control -> Flight

1. **Synthesize** (`src/sousvide/synthesize/`): Generate training data by rolling out expert trajectories in FiGS with domain randomization (mass/thrust variations, state perturbations). Outputs state/control/image tuples.

2. **Instruct** (`src/sousvide/instruct/`): Train neural network policies on synthesized data. `train_policy.py` orchestrates training with MSE-based loss. Data loaded via `ObservationData` dataset class from `.pt` files.

3. **Control** (`src/sousvide/control/`): Neural network policy architectures. `Policy` (nn.Module) wraps feature extractors + command networks. `Pilot` extends FiGS `BaseController` for simulator integration. `VLAPolicy` wraps external VLA servers (OpenPI) via websocket.

4. **Flight** (`src/sousvide/flight/`): Real-world deployment (ROS-based) and FiGS evaluation. Computes metrics: trajectory tracking error (TTE), proximity percentile (PP), inference Hz.

### Falsification System (`src/sousvide/falsification/`)

The main research contribution. Systematically stress-tests VLA policies:

- **`config.py`**: `GATE_PRESETS`, `DEFAULT_CONFIG`, `apply_gate_preset()`, `load_config()` — shared configuration for all falsification scripts
- **`orchestrator.py`**: `FalsificationOrchestrator` runs episodes: reset -> apply perturbations -> run VLA -> detect failure -> plan recovery
- **`perturbations.py`**: Three perturbation surfaces — **Action** (noise/bias/scale on thrust+rates), **Observation** (image/state/camera noise), **Environment** (shift/scale/opacity of Gaussians via Splat-MOVER)
- **`failure_detector.py`**: Pluggable `SafetyCriterion` voting — bounds, velocity, tilt, proximity collision
- **`splatnav_recovery.py`**: Plans collision-free recovery trajectories from failure states using SplatNav

### Git Submodules

- **FiGS/** (`FiGS`): Flight-in-Gaussian-Splats simulator — ACADOS dynamics, GSplat rendering, MPC trajectory tracking, MinTimeSnap trajectory generation
- **external/splatnav**: Collision-aware A* + spline path planning over Gaussian splat scenes
- **external/Splat-MOVER**: Gaussian splat scene editing (used for environment perturbations)

### Coordinate Frames

`src/sousvide/utilities/coordinate_transform.py` (canonical location) provides `CoordinateTransformer` for converting between:
- **FiGS/NED**: North-East-Down (dynamics reference)
- **MOCAP/Z-up**: Motion capture from real experiments
- **COLMAP**: Camera-centric reconstruction frame
- **Nerfstudio**: Used internally by SplatNav

Alignment data loaded from JSON files in `data/alignment/left_gate/` and `data/alignment/right_gate/`. Getting coordinate chains right is critical — failures here propagate silently.

## Key Entry Points

| Script | Purpose |
|--------|---------|
| `run_falsification.py` | Main falsification campaign — configures FiGS, VLA, perturbations, safety criteria, runs N episodes |
| `view_falsification_trajectories.py` | Visualize falsification results in Nerfstudio viewer |
| `scripts/recover_falsification_episode.py` | Post-hoc recovery planning on saved episodes |
| `scripts/render_falsification_cameras.py` | Render camera views from saved episodes |
| `scripts/deploy/policy_inference.py` | Real-time ROS node for physical drone deployment |
| `scripts/visualization/ns_viewer_with_trajectories.py` | Visualize trajectories in Nerfstudio viewer |

## Configuration

- **`src/sousvide/falsification/config.py`**: Gate presets and default falsification config
- **`configs/falsification/`**: YAML overrides for falsification runs (config_gate_only.yaml, etc.)
- **`configs/pilots/`**: Named pilot profiles (network architecture, hyperparameters) — e.g., Iceman.json, Maverick.json
- **`configs/courses/`**: Trajectory waypoints and external forces
- **`configs/frames/`**: Drone specs (mass, thrust coefficient, rotors)
- **`configs/methods/`**: Simulation config (duration, rate, frequency)
- **`configs/nnio/`**: Neural network I/O mappings

All FiGS configs loaded via `figs.utilities.config_helper.get_config()`.

## Data Layout

```
data/alignment/left_gate/              # COLMAP-to-MOCAP Sim(3) transforms
data/alignment/right_gate/
data/colmap/left_gate/                 # COLMAP reconstructions & trained GSplats
data/colmap/right_gate/
cohorts/<cohort>/rollout_data/...      # Training data
falsification_results/<gate>/episodes/ # Falsification output
```

## Network Architecture Types

Defined in `src/sousvide/control/networks/`:
- **SIFU**: Sequence Into Features (Unified) — MLPs processing history sequences
- **SVNet**: State-Vector command network
- **DNNet**: Dynamic network
- **Pave**: Probabilistic network
- **DINO**: Vision transformer feature extractor

Instantiated via factory pattern in `network_factory.py`.
