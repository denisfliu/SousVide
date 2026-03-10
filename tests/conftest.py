"""Shared fixtures for the falsification test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded RandomState for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def identity_quat():
    """Identity quaternion [qx, qy, qz, qw]."""
    return np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture
def sample_state(identity_quat):
    """A 10-element state: [px,py,pz, vx,vy,vz, qx,qy,qz,qw]."""
    return np.concatenate([
        np.array([1.0, 2.0, -1.5]),   # position
        np.array([0.5, -0.3, 0.1]),    # velocity
        identity_quat,
    ])


@pytest.fixture
def sample_action():
    """A 4-element control: [thrust, wx, wy, wz]."""
    return np.array([-0.5, 0.1, -0.1, 0.05])
