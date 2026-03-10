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

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock* ./

# Copy submodule manifests so editable installs resolve
COPY FiGS/pyproject.toml FiGS/setup.py FiGS/setup.cfg FiGS/README.md FiGS/
COPY FiGS/src/ FiGS/src/
COPY FiGS/acados/ FiGS/acados/
COPY FiGS/Hierarchical-Localization/ FiGS/Hierarchical-Localization/
COPY external/splatnav/ external/splatnav/
COPY external/Splat-MOVER/ external/Splat-MOVER/

# Install dependencies (builds tinycudann from source)
ENV CC=gcc-11 CXX=g++-11 CUDA_HOME=/usr/local/cuda
RUN uv sync --no-dev

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY run_falsification.py .
COPY view_falsification_trajectories.py .
COPY scripts/ scripts/

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
COPY --from=builder /app/FiGS /app/FiGS
COPY --from=builder /app/external /app/external
COPY --from=builder /app/run_falsification.py /app/
COPY --from=builder /app/view_falsification_trajectories.py /app/
COPY --from=builder /app/scripts /app/scripts

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:/app/FiGS/src:/app/external/splatnav"
ENV LD_LIBRARY_PATH="/app/FiGS/acados/lib"
ENV ACADOS_SOURCE_DIR="/app/FiGS/acados"

ENTRYPOINT ["python", "run_falsification.py"]
CMD ["--help"]
