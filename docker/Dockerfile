# Multi-stage Dockerfile for VLA Falsification
# Stage 1: build environment (needs nvcc for tinycudann)
# Stage 2: slim runtime image

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git gcc-11 g++-11 cmake ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make gcc-11 the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy everything needed for uv sync (manifests + source for editable installs)
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY external/FiGS/pyproject.toml external/FiGS/README.md external/FiGS/
COPY external/FiGS/src/ external/FiGS/src/
COPY external/FiGS/acados/ external/FiGS/acados/
COPY external/FiGS/Hierarchical-Localization/ external/FiGS/Hierarchical-Localization/
COPY external/splatnav/ external/splatnav/
COPY external/Splat-MOVER/ external/Splat-MOVER/

# Install dependencies (builds tinycudann from source)
# tinycudann has no-build-isolation: needs torch+setuptools pre-installed.
# TCNN_CUDA_ARCHITECTURES required because no GPU available during build.
ENV CC=gcc-11 CXX=g++-11 CUDA_HOME=/usr/local/cuda
ENV TCNN_CUDA_ARCHITECTURES="89"
RUN uv venv \
 && uv pip install "setuptools<71" wheel numpy \
 && uv pip install torch --index-url https://download.pytorch.org/whl/cu121 \
 && uv sync --no-dev

# Copy remaining project files
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY data/alignment/ data/alignment/
COPY run_falsification.py .
COPY view_falsification_trajectories.py .
COPY scripts/ scripts/

# FiGS config_helper resolves configs relative to external/
RUN ln -s /app/configs /app/external/configs

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built virtual environment and project files from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/artifacts /app/artifacts
COPY --from=builder /app/data/alignment /app/data/alignment
COPY --from=builder /app/external /app/external
COPY --from=builder /app/run_falsification.py /app/
COPY --from=builder /app/view_falsification_trajectories.py /app/
COPY --from=builder /app/scripts /app/scripts

# FiGS config_helper resolves configs relative to external/
RUN ln -s /app/configs /app/external/configs

ENV PATH="/app/.venv/bin:$PATH"
ENV TORCH_COMPILE_DISABLE=1
ENV PYTHONPATH="/app/src:/app/external/FiGS/src:/app/external/splatnav"
ENV LD_LIBRARY_PATH="/app/external/FiGS/acados/lib"
ENV ACADOS_SOURCE_DIR="/app/external/FiGS/acados"

ENTRYPOINT ["python", "run_falsification.py"]
CMD ["--help"]
