# SousVide

**Scene Understanding via Synthesized Visual Inertial Data from Experts**

A research framework for systematic falsification and recovery of Vision-Language-Action (VLA) control policies for drones in photorealistic Gaussian Splatting environments. Built on top of the FiGS flight simulator.

## Quick Start (Docker)

```bash
# Build the image (requires NVIDIA GPU + nvidia-container-toolkit)
docker build -f docker/Dockerfile -t sousvide .

# Run a falsification campaign
docker run --gpus all sousvide --gate left_gate --num-episodes 10

# Or use docker compose
cd docker && docker compose up
```

## Quick Start (Development)

### Prerequisites

- CUDA 11.8 toolkit (local install for compiling tiny-cuda-nn)
- COLMAP (`sudo apt install colmap`)
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/StanfordMSL/SousVide.git
cd SousVide
git submodule update --recursive --init

# Build ACADOS
cd FiGS/acados && mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON .. && make install -j4
cd ../../..

# Add to ~/.zshrc or ~/.bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$(pwd)/FiGS/acados/lib"
export ACADOS_SOURCE_DIR="$(pwd)/FiGS/acados"
export CUDA_HOME=$HOME/.local/cuda-11.8

# Install Python environment
uv sync
source .venv/bin/activate
```

### CUDA 11.8 Toolkit (local install)

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit \
    --installpath=$HOME/.local/cuda-11.8 --no-man-page

# Add to shell config:
export CUDA_HOME=$HOME/.local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
```

### Download GSplats

Place COLMAP data under `data/colmap/left_gate/` and `data/colmap/right_gate/`.
Place alignment data under `data/alignment/left_gate/` and `data/alignment/right_gate/`.

## Usage

```bash
# Run a falsification campaign
python run_falsification.py --gate left_gate --num-episodes 50

# Visualize results
python view_falsification_trajectories.py --results-dir falsification_results/left_gate

# Post-hoc recovery planning
python scripts/recover_falsification_episode.py --results-dir falsification_results/left_gate --episode 3

# Render camera views from an episode
python scripts/render_falsification_cameras.py --results-dir falsification_results/left_gate --episode 0
```

## Directory Structure

```
SousVide/
├── run_falsification.py              # Main entry point
├── view_falsification_trajectories.py # Primary visualization tool
├── src/sousvide/
│   ├── control/                       # Policy architectures & VLA wrapper
│   ├── falsification/                 # Orchestrator, perturbations, failure detection
│   │   ├── config.py                  # Gate presets & default configuration
│   │   ├── orchestrator.py            # Episode runner
│   │   ├── perturbations.py           # Action/observation/environment perturbations
│   │   ├── failure_detector.py        # Safety criteria
│   │   └── splatnav_recovery.py       # Collision-free recovery planning
│   ├── utilities/
│   │   └── coordinate_transform.py    # NED/MOCAP/COLMAP/Nerfstudio conversions
│   ├── synthesize/                    # Training data generation
│   ├── instruct/                      # Policy training
│   └── flight/                        # Real-world deployment & evaluation
├── scripts/
│   ├── recover_falsification_episode.py
│   ├── render_falsification_cameras.py
│   ├── debug/                         # Gate perturbation debugging tools
│   ├── demo/                          # Trajectory generation demos
│   ├── deploy/                        # ROS policy inference node
│   └── visualization/                 # Nerfstudio viewer with trajectories
├── configs/
│   ├── falsification/                 # YAML overrides (config_gate_only.yaml, etc.)
│   ├── pilots/                        # Named pilot profiles
│   ├── courses/                       # Trajectory waypoints
│   ├── frames/                        # Drone specifications
│   └── methods/                       # Simulation settings
├── data/
│   ├── alignment/                     # COLMAP-to-MOCAP Sim(3) transforms
│   └── colmap/                        # COLMAP reconstructions & trained GSplats
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/                             # pytest suite (114 tests)
├── FiGS/                              # Flight-in-Gaussian-Splats simulator (submodule)
└── external/
    ├── splatnav/                      # Collision-aware path planning (submodule)
    └── Splat-MOVER/                   # GSplat scene editing (submodule)
```

## Configuration

Falsification runs are configured via:
1. Built-in defaults in `src/sousvide/falsification/config.py`
2. Gate presets (`--gate left_gate` or `--gate right_gate`)
3. YAML overrides (`--config configs/falsification/config_gate_only.yaml`)
4. CLI arguments (`--num-episodes`, `--seed`, etc.)

## Testing

```bash
PYTHONPATH="src:.:$PYTHONPATH" python -m pytest tests/ -v
```

## Real-World Deployment

Deploy SousVide policies on an [MSL Drone](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware):

```bash
python scripts/deploy/policy_inference.py
```
