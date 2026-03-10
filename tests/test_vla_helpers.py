"""Tests for VLA policy helper functions."""

import numpy as np
import pytest

from sousvide.control.vla_policy import _waypoints_to_tXUd


class TestWaypointsToTXUd:
    def test_basic_shape(self):
        state = np.concatenate([
            np.array([0, 0, -1.0]),  # pos
            np.zeros(3),              # vel
            np.array([0, 0, 0, 1.0]),  # quat
        ])
        waypoints = np.array([
            [0.1, 0, -1.0],
            [0.2, 0, -1.0],
            [0.3, 0, -1.0],
        ])
        result = _waypoints_to_tXUd(state, waypoints, hz=10, m=0.5, kt=7.0, n_rotors=4)
        assert result.ndim == 2
        assert result.shape[1] == 15
        # At least 30 rows (min_pts)
        assert result.shape[0] >= 30

    def test_time_column(self):
        state = np.zeros(10)
        state[9] = 1.0
        waypoints = np.array([[1, 0, 0], [2, 0, 0]])
        result = _waypoints_to_tXUd(state, waypoints, hz=10, m=1.0, kt=1.0, n_rotors=4)
        dt = 1.0 / 10
        for i in range(result.shape[0]):
            assert result[i, 0] == pytest.approx(i * dt)

    def test_first_position_is_current(self):
        state = np.zeros(10)
        state[:3] = [5.0, 3.0, -1.0]
        state[9] = 1.0
        waypoints = np.array([[5.1, 3.0, -1.0]])
        result = _waypoints_to_tXUd(state, waypoints, hz=10, m=1.0, kt=1.0, n_rotors=4)
        np.testing.assert_allclose(result[0, 1:4], [5.0, 3.0, -1.0])

    def test_hover_thrust(self):
        state = np.zeros(10)
        state[9] = 1.0
        waypoints = np.array([[0, 0, 0]])
        m, kt, n_rotors, g = 0.5, 7.0, 4, 9.81
        result = _waypoints_to_tXUd(state, waypoints, hz=10, m=m, kt=kt, n_rotors=n_rotors, g=g)
        expected_thrust = -(m * g) / (n_rotors * kt)
        assert result[0, 11] == pytest.approx(expected_thrust)

    def test_padding_to_min_pts(self):
        state = np.zeros(10)
        state[9] = 1.0
        waypoints = np.array([[1, 0, 0]])
        result = _waypoints_to_tXUd(state, waypoints, hz=10, m=1.0, kt=1.0, n_rotors=4)
        assert result.shape[0] >= 30
        # Padded positions should be the last waypoint
        np.testing.assert_allclose(result[-1, 1:4], waypoints[-1])

    def test_velocity_zero_in_padding(self):
        state = np.zeros(10)
        state[9] = 1.0
        waypoints = np.array([[1, 0, 0], [2, 0, 0]])
        result = _waypoints_to_tXUd(state, waypoints, hz=10, m=1.0, kt=1.0, n_rotors=4)
        # Last row (padded) should have zero velocity
        np.testing.assert_allclose(result[-1, 4:7], [0, 0, 0])
