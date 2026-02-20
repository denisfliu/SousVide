# SOUS VIDE

## Installation
1) Clone repository and load the submodules.
```
git clone https://github.com/StanfordMSL/SousVide.git
git submodule update --recursive --init
```
2) Build ACADOS locally.
```
# Navigate to acados folder
cd <repository-path>/SousVide/FiGS/acados/

# Compile
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

# Add acados paths to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```
3) Install COLMAP (requires apt).
```
sudo apt install colmap
```

4) Install the CUDA 11.8 toolkit locally (no root / system-wide install required).
The runfile installer supports a custom `--installpath`, so everything stays inside your home directory.
```
# Download the CUDA 11.8 runfile
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run

# Install toolkit only to ~/.local/cuda-11.8  (no sudo, no driver install)
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit \
    --installpath=$HOME/.local/cuda-11.8 \
    --no-man-page

# Add to your shell config (~/.zshrc or ~/.bashrc) and reload
export CUDA_HOME=$HOME/.local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
The CUDA toolkit is only needed to compile `tiny-cuda-nn`. Pre-built PyTorch wheels do not require it.

5) Set up the Python environment with [uv](https://docs.astral.sh/uv/).

`uv sync` installs everything in one shot: PyTorch (CUDA 11.8 pre-built wheels), nerfstudio, FiGS, Splat-Nav, hloc, and all other dependencies. `tiny-cuda-nn` is compiled from source during `uv sync`, so the CUDA toolkit from step 4 must be on your PATH first.
```
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the repository root
cd <repository-path>/SousVide/

# Create a virtual environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```
6) Download Example GSplats
```
# Navigate to gsplats parent folder
cd <repository-path>/SousVide/

# Download the zip-ed file below and unpack the contents (capture and workspace) into the gsplats folder
https://drive.google.com/file/d/1kW5dzsfD3rbRA3RIQDyJPG6_UJaO9ALP/view
```

## Run SOUS VIDE Examples
Check out the notebook examples in the notebooks folder:
  1. <b>figs_examples</b>: Example code for generating GSplats and executing trajectories within them (using FiGS).
  2. <b>sous_vide_examples</b>: Use this notebook to try out two of the policies generated in the paper.


## [COMING SOON: Oct 2025] Deploy SOUS VIDE in the Real World
Deploy SOUS VIDE policies on an [MSL Drone](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware). Tutorial and code coming soon!