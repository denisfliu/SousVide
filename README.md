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
3) Set up conda environment (in the main directory)
```
# Navigate to environment config location
cd <repository-path>/SousVide/

# Create and activate
conda env create -f environment_x86.yml
conda activate kitchen
```
4) Do the remainder of the pip installs
```
# Because tiny-cuda-nn does not play well with pip 25 (which is used by the transformers package)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-use-pep517

# Then install the remainder of the packages
pip install -r requirements-rest.txt

# Force install the latest version of timm (for transformers package)
pip install timm==1.0.15

# (Maybe do this). This keeps the versioning more pallatable.
pip install casadi==3.6.7

# Note that pip will throw an error stating that nerfstudio requires 0.6.7 while transformers needs a later version (1.0.15 at time of writing). Ignore this as, as shown in [here](https://github.com/nerfstudio-project/nerfstudio/pull/3637), timm is no longer used in nerfstudio.
```
5) Download Example GSplats
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