#!/bin/bash
# Convenience wrapper — sets up the environment and forwards all arguments.
# Usage: ./run.sh run_falsification.py --gate left_gate --num-episodes 1
#        ./run.sh view_falsification_trajectories.py --results-dir falsification_results/left_gate_v4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/external/FiGS/src:$SCRIPT_DIR/external/splatnav:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SCRIPT_DIR/external/FiGS/acados/lib:${LD_LIBRARY_PATH:-}"
export ACADOS_SOURCE_DIR="$SCRIPT_DIR/external/FiGS/acados"
export TORCH_COMPILE_DISABLE=1
export CC=gcc-11
export CXX=g++-11

exec python "$@"
