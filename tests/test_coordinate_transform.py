"""Tests for coordinate transformation utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sousvide.utilities.coordinate_transform import (
    CoordinateTransformer,
    build_camera_transforms,
    build_figs_to_nerf_transform,
    convert_ned_to_zup,
    convert_zup_to_ned,
    create_transformer_for_scene,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sim3_data():
    """Realistic Sim(3) parameters for testing."""
    # Rotation: 30° around z-axis
    angle = np.radians(30)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1],
    ])
    return {
        "scale": 2.5,
        "R": R.tolist(),
        "t": [0.1, -0.2, 0.3],
    }


@pytest.fixture
def alignment_dir(sim3_data, tmp_path):
    """Temporary alignment directory with sim3 JSON."""
    sim3_path = tmp_path / "colmap_to_mocap_sim3.json"
    sim3_path.write_text(json.dumps(sim3_data))
    return str(tmp_path)


@pytest.fixture
def transformer(alignment_dir):
    return CoordinateTransformer(alignment_dir)


# ===================================================================
# CoordinateTransformer
# ===================================================================


class TestCoordinateTransformer:
    def test_loading(self, transformer, sim3_data):
        assert transformer.s == sim3_data["scale"]
        np.testing.assert_allclose(transformer.R, sim3_data["R"])
        np.testing.assert_allclose(transformer.t, sim3_data["t"])

    def test_inverse_scale(self, transformer, sim3_data):
        assert transformer.s_inv == pytest.approx(1.0 / sim3_data["scale"])

    def test_inverse_rotation(self, transformer):
        np.testing.assert_allclose(
            transformer.R_inv, transformer.R.T, atol=1e-10
        )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CoordinateTransformer(str(tmp_path))

    def test_position_roundtrip(self, transformer):
        pos_mocap = np.array([1.0, -0.5, 0.8])
        pos_colmap = transformer.mocap_to_colmap_position(pos_mocap)

        # Manual inverse: pos_mocap = s * R @ pos_colmap + t
        pos_back = transformer.s * (transformer.R @ pos_colmap) + transformer.t
        np.testing.assert_allclose(pos_back, pos_mocap, atol=1e-10)

    def test_pose_roundtrip(self, transformer):
        T_mocap = np.eye(4)
        T_mocap[:3, :3] = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        T_mocap[:3, 3] = [1.5, -0.3, 2.0]

        T_colmap = transformer.mocap_to_colmap_pose(T_mocap)
        T_back = transformer.colmap_to_mocap_pose(T_colmap)

        np.testing.assert_allclose(T_back[:3, :3], T_mocap[:3, :3], atol=1e-10)
        np.testing.assert_allclose(T_back[:3, 3], T_mocap[:3, 3], atol=1e-10)

    def test_identity_transform(self, tmp_path):
        """Identity Sim(3) should not change positions."""
        sim3 = {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0]}
        (tmp_path / "colmap_to_mocap_sim3.json").write_text(json.dumps(sim3))
        t = CoordinateTransformer(str(tmp_path))

        pos = np.array([3.0, -1.0, 7.0])
        np.testing.assert_allclose(
            t.mocap_to_colmap_position(pos), pos, atol=1e-10
        )

    def test_get_transformation_info(self, transformer, sim3_data):
        info = transformer.get_transformation_info()
        assert info["colmap_to_mocap"]["scale"] == sim3_data["scale"]
        assert info["mocap_to_colmap"]["scale"] == pytest.approx(1.0 / sim3_data["scale"])


# ===================================================================
# Scene factory (only if alignment data is present)
# ===================================================================


class TestCreateTransformerForScene:
    def test_unknown_scene_raises(self):
        with pytest.raises(ValueError, match="Unknown scene"):
            create_transformer_for_scene("nonexistent_scene")

    def test_left_gate_loads(self):
        """Only runs if alignment data exists locally."""
        try:
            t = create_transformer_for_scene("left_gate")
            assert t.s > 0
        except FileNotFoundError:
            pytest.skip("Left gate alignment data not available")

    def test_right_gate_loads(self):
        """Only runs if alignment data exists locally."""
        try:
            t = create_transformer_for_scene("right_gate")
            assert t.s > 0
        except FileNotFoundError:
            pytest.skip("Right gate alignment data not available")


# ===================================================================
# NED ↔ Z-up conversions
# ===================================================================


class TestNedConversions:
    @pytest.mark.parametrize("perm", [0, 2, 5])
    def test_roundtrip(self, perm):
        pos_zup = np.array([1.5, -0.7, 2.3])
        ned = convert_zup_to_ned(pos_zup, perm=perm)
        back = convert_ned_to_zup(ned, perm=perm)
        np.testing.assert_allclose(back, pos_zup, atol=1e-10)

    def test_perm5_signs(self):
        pos = np.array([1.0, 2.0, 3.0])
        ned = convert_zup_to_ned(pos, perm=5)
        np.testing.assert_allclose(ned, [1.0, -2.0, -3.0])

    def test_origin(self):
        for perm in [0, 2, 5]:
            ned = convert_zup_to_ned(np.zeros(3), perm=perm)
            np.testing.assert_allclose(ned, [0, 0, 0])


# ===================================================================
# Camera transforms
# ===================================================================


class TestBuildCameraTransforms:
    def test_returns_two_4x4(self):
        fwd, dwn = build_camera_transforms()
        assert fwd.shape == (4, 4)
        assert dwn.shape == (4, 4)

    def test_homogeneous(self):
        fwd, dwn = build_camera_transforms()
        np.testing.assert_allclose(fwd[3, :], [0, 0, 0, 1])
        np.testing.assert_allclose(dwn[3, :], [0, 0, 0, 1])

    def test_rotation_orthogonal(self):
        fwd, dwn = build_camera_transforms()
        R_fwd = fwd[:3, :3]
        R_dwn = dwn[:3, :3]
        np.testing.assert_allclose(R_fwd @ R_fwd.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(R_dwn @ R_dwn.T, np.eye(3), atol=1e-10)
