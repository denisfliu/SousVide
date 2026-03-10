"""Tests for NED <-> Z-up coordinate conversions."""

import numpy as np
import pytest

from vla_falsification.utilities.coordinate_transform import convert_ned_to_zup, convert_zup_to_ned
from vla_falsification.falsification.config import convert_to_ned, convert_from_ned_to_zup


class TestConvertZupToNed:
    def test_perm5_roundtrip(self):
        pos_zup = np.array([1.0, -0.5, 0.8])
        ned = convert_zup_to_ned(pos_zup, perm=5)
        back = convert_ned_to_zup(ned, perm=5)
        np.testing.assert_allclose(back, pos_zup, atol=1e-10)

    def test_perm0_roundtrip(self):
        pos_zup = np.array([2.0, 1.0, 3.0])
        ned = convert_zup_to_ned(pos_zup, perm=0)
        back = convert_ned_to_zup(ned, perm=0)
        np.testing.assert_allclose(back, pos_zup, atol=1e-10)

    def test_perm2_roundtrip(self):
        pos_zup = np.array([-1.5, 0.7, 2.3])
        ned = convert_zup_to_ned(pos_zup, perm=2)
        back = convert_ned_to_zup(ned, perm=2)
        np.testing.assert_allclose(back, pos_zup, atol=1e-10)

    def test_perm5_z_negated(self):
        pos_zup = np.array([1.0, 2.0, 3.0])
        ned = convert_zup_to_ned(pos_zup, perm=5)
        np.testing.assert_allclose(ned, [1.0, -2.0, -3.0])

    def test_perm0_z_negated(self):
        pos_zup = np.array([1.0, 2.0, 3.0])
        ned = convert_zup_to_ned(pos_zup, perm=0)
        np.testing.assert_allclose(ned, [1.0, 2.0, -3.0])

    def test_origin(self):
        for perm in [0, 2, 5]:
            ned = convert_zup_to_ned(np.zeros(3), perm=perm)
            np.testing.assert_allclose(ned, [0, 0, 0])


class TestConfigAliases:
    """Ensure vla_falsification.falsification.config aliases match the canonical functions."""

    def test_convert_to_ned_matches(self):
        pos = np.array([1.0, -0.5, 0.8])
        np.testing.assert_allclose(
            convert_to_ned(pos, perm=5),
            convert_zup_to_ned(pos, perm=5),
        )

    def test_convert_from_ned_matches(self):
        ned = np.array([1.0, 0.5, -0.8])
        np.testing.assert_allclose(
            convert_from_ned_to_zup(ned, perm=5),
            convert_ned_to_zup(ned, perm=5),
        )
