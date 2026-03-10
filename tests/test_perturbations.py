"""Tests for the perturbation engine (sousvide.falsification.perturbations)."""

import numpy as np
import pytest
import torch

from sousvide.falsification.perturbations import (
    ActionBias,
    ActionBiasConfig,
    ActionNoise,
    ActionNoiseConfig,
    ActionScale,
    ActionScaleConfig,
    BrightnessShift,
    BrightnessShiftConfig,
    CameraPoseNoise,
    CameraPoseNoiseConfig,
    GaussianOpacity,
    GaussianOpacityConfig,
    GaussianScale,
    GaussianScaleConfig,
    GaussianShift,
    GaussianShiftConfig,
    ImageNoise,
    ImageNoiseConfig,
    ImageOcclusion,
    ImageOcclusionConfig,
    PerturbationStack,
    PerturbationSuite,
    StateEstimateNoise,
    StateEstimateNoiseConfig,
    build_perturbation_suite,
    _quat_multiply,
    _quat_multiply_wxyz_torch,
)


# ===================================================================
# Action perturbations
# ===================================================================


class TestActionNoise:
    def test_apply_changes_action(self, rng, sample_action):
        p = ActionNoise(ActionNoiseConfig(std=np.array([1.0, 1.0, 1.0, 1.0])))
        p.reset(rng)
        result = p.apply(sample_action)
        assert result.shape == sample_action.shape
        assert not np.allclose(result, sample_action)

    def test_zero_noise_is_identity(self, rng, sample_action):
        p = ActionNoise(ActionNoiseConfig(
            mean=np.zeros(4), std=np.zeros(4)
        ))
        p.reset(rng)
        result = p.apply(sample_action)
        np.testing.assert_array_equal(result, sample_action)

    def test_default_config(self, rng, sample_action):
        p = ActionNoise()
        p.reset(rng)
        result = p.apply(sample_action)
        assert result.shape == (4,)

    def test_reproducibility(self, sample_action):
        p = ActionNoise(ActionNoiseConfig(std=np.ones(4)))
        p.reset(np.random.RandomState(123))
        r1 = p.apply(sample_action.copy())
        p.reset(np.random.RandomState(123))
        r2 = p.apply(sample_action.copy())
        np.testing.assert_array_equal(r1, r2)


class TestActionBias:
    def test_constant_across_steps(self, rng, sample_action):
        p = ActionBias(ActionBiasConfig(
            bias_range=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
        ))
        p.reset(rng)
        r1 = p.apply(sample_action.copy())
        r2 = p.apply(sample_action.copy())
        np.testing.assert_array_equal(r1, r2)

    def test_bias_within_range(self, rng, sample_action):
        bias_range = np.array([[-0.5, 0.5], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
        p = ActionBias(ActionBiasConfig(bias_range=bias_range))
        p.reset(rng)
        diff = p.apply(sample_action) - sample_action
        for i in range(4):
            assert bias_range[i, 0] <= diff[i] <= bias_range[i, 1]

    def test_changes_on_reset(self, sample_action):
        p = ActionBias()
        p.reset(np.random.RandomState(1))
        r1 = p.apply(sample_action.copy())
        p.reset(np.random.RandomState(2))
        r2 = p.apply(sample_action.copy())
        assert not np.allclose(r1, r2)


class TestActionScale:
    def test_scaling(self, rng, sample_action):
        p = ActionScale(ActionScaleConfig(scale_range=(2.0, 2.0)))
        p.reset(rng)
        result = p.apply(sample_action)
        np.testing.assert_allclose(result, sample_action * 2.0)

    def test_scale_within_range(self, rng, sample_action):
        p = ActionScale(ActionScaleConfig(scale_range=(0.5, 1.5)))
        p.reset(rng)
        result = p.apply(sample_action)
        ratio = result / sample_action
        assert np.all(ratio >= 0.5 - 1e-10) and np.all(ratio <= 1.5 + 1e-10)


# ===================================================================
# Observation perturbations
# ===================================================================


class TestImageNoise:
    def test_output_uint8(self, rng):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        p = ImageNoise(ImageNoiseConfig(std=20.0))
        p.reset(rng)
        result = p.apply(img)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_clipping(self, rng):
        img = np.full((64, 64, 3), 250, dtype=np.uint8)
        p = ImageNoise(ImageNoiseConfig(std=100.0))
        p.reset(rng)
        result = p.apply(img)
        assert result.max() <= 255
        assert result.min() >= 0

    def test_zero_noise_preserves_image(self, rng):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        p = ImageNoise(ImageNoiseConfig(std=0.0))
        p.reset(rng)
        result = p.apply(img)
        np.testing.assert_array_equal(result, img)


class TestImageOcclusion:
    def test_patches_applied(self, rng):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        p = ImageOcclusion(ImageOcclusionConfig(num_patches=3, color=(0, 0, 0)))
        p.reset(rng)
        result = p.apply(img)
        assert np.any(result == 0), "Occlusion patches should be visible"

    def test_original_not_modified(self, rng):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        original = img.copy()
        p = ImageOcclusion()
        p.reset(rng)
        p.apply(img)
        np.testing.assert_array_equal(img, original)


class TestBrightnessShift:
    def test_shift_applied(self, rng):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        p = BrightnessShift(BrightnessShiftConfig(shift_range=(30.0, 30.0)))
        p.reset(rng)
        result = p.apply(img)
        assert np.all(result == 158)

    def test_negative_shift_clamps(self, rng):
        img = np.full((32, 32, 3), 10, dtype=np.uint8)
        p = BrightnessShift(BrightnessShiftConfig(shift_range=(-50.0, -50.0)))
        p.reset(rng)
        result = p.apply(img)
        assert np.all(result == 0)


class TestCameraPoseNoise:
    def test_perturbs_transform(self, rng):
        T = np.eye(4)
        p = CameraPoseNoise(CameraPoseNoiseConfig(
            position_std=0.1, rotation_std=0.05
        ))
        p.reset(rng)
        result = p.apply(T)
        assert result.shape == (4, 4)
        assert not np.allclose(result, T)

    def test_rotation_stays_orthogonal(self, rng):
        T = np.eye(4)
        p = CameraPoseNoise(CameraPoseNoiseConfig(
            position_std=0.1, rotation_std=0.1
        ))
        p.reset(rng)
        result = p.apply(T)
        R = result[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_original_not_modified(self, rng):
        T = np.eye(4)
        original = T.copy()
        p = CameraPoseNoise()
        p.reset(rng)
        p.apply(T)
        np.testing.assert_array_equal(T, original)


class TestStateEstimateNoise:
    def test_perturbs_state(self, rng, sample_state):
        p = StateEstimateNoise(StateEstimateNoiseConfig(
            position_std=0.1, velocity_std=0.1, orientation_std=0.05
        ))
        p.reset(rng)
        result = p.apply(sample_state)
        assert result.shape == sample_state.shape
        assert not np.allclose(result[:3], sample_state[:3])

    def test_quaternion_normalized(self, rng, sample_state):
        p = StateEstimateNoise(StateEstimateNoiseConfig(orientation_std=0.1))
        p.reset(rng)
        result = p.apply(sample_state)
        q_norm = np.linalg.norm(result[6:10])
        np.testing.assert_allclose(q_norm, 1.0, atol=1e-10)

    def test_original_not_modified(self, rng, sample_state):
        original = sample_state.copy()
        p = StateEstimateNoise()
        p.reset(rng)
        p.apply(sample_state)
        np.testing.assert_array_equal(sample_state, original)


# ===================================================================
# Environment perturbations
# ===================================================================


class TestGaussianShift:
    def test_shifts_subset(self, rng):
        means = torch.randn(1000, 3)
        p = GaussianShift(GaussianShiftConfig(fraction=0.1, shift_std=0.5))
        p.reset(rng)
        result = p.apply(means)
        n_changed = (result != means).any(dim=1).sum().item()
        assert 50 <= n_changed <= 150  # ~100 expected

    def test_restore(self, rng):
        means = torch.randn(500, 3)
        p = GaussianShift(GaussianShiftConfig(fraction=0.1, shift_std=0.5))
        p.reset(rng)
        perturbed = p.apply(means)
        restored = p.restore(perturbed)
        torch.testing.assert_close(restored, means)

    def test_consistent_across_calls(self, rng):
        means = torch.randn(100, 3)
        p = GaussianShift(GaussianShiftConfig(fraction=0.5))
        p.reset(rng)
        r1 = p.apply(means)
        r2 = p.apply(means)
        # Same indices and shifts should be applied
        torch.testing.assert_close(r1, r2)


class TestGaussianScale:
    def test_scales_subset(self, rng):
        scales = torch.ones(500, 3)
        p = GaussianScale(GaussianScaleConfig(fraction=0.1, scale_range=(2.0, 2.0)))
        p.reset(rng)
        result = p.apply(scales)
        n_changed = (result != scales).any(dim=1).sum().item()
        assert n_changed > 0

    def test_restore(self, rng):
        scales = torch.randn(500, 3).abs() + 0.1
        p = GaussianScale(GaussianScaleConfig(fraction=0.2))
        p.reset(rng)
        perturbed = p.apply(scales)
        restored = p.restore(perturbed)
        torch.testing.assert_close(restored, scales)


class TestGaussianOpacity:
    def test_sets_opacity(self, rng):
        opacities = torch.full((200,), 0.5)
        p = GaussianOpacity(GaussianOpacityConfig(fraction=0.1, opacity_range=(0.0, 0.0)))
        p.reset(rng)
        result = p.apply(opacities)
        n_zeroed = (result == 0.0).sum().item()
        assert n_zeroed > 0

    def test_restore(self, rng):
        opacities = torch.rand(200)
        p = GaussianOpacity(GaussianOpacityConfig(fraction=0.3))
        p.reset(rng)
        perturbed = p.apply(opacities)
        restored = p.restore(perturbed)
        torch.testing.assert_close(restored, opacities)


# ===================================================================
# Composition
# ===================================================================


class TestPerturbationStack:
    def test_sequential_application(self, rng, sample_action):
        stack = PerturbationStack([
            ActionBias(ActionBiasConfig(
                bias_range=np.array([[0.1, 0.1]] * 4)
            )),
            ActionScale(ActionScaleConfig(scale_range=(2.0, 2.0))),
        ])
        stack.reset(rng)
        result = stack.apply(sample_action)
        expected = (sample_action + 0.1) * 2.0
        np.testing.assert_allclose(result, expected)

    def test_empty_stack(self, sample_action):
        stack = PerturbationStack()
        stack.reset(np.random.RandomState(0))
        result = stack.apply(sample_action)
        np.testing.assert_array_equal(result, sample_action)

    def test_len(self):
        stack = PerturbationStack([ActionNoise(), ActionBias()])
        assert len(stack) == 2


class TestPerturbationSuite:
    def test_reset_all(self):
        suite = PerturbationSuite(
            action=PerturbationStack([ActionNoise()]),
            observation_image=PerturbationStack([ImageNoise()]),
        )
        suite.reset_all(seed=42)
        # Should not raise

    def test_default_empty(self):
        suite = PerturbationSuite()
        assert len(suite.action) == 0
        assert len(suite.observation_image) == 0
        assert len(suite.environment_means) == 0


# ===================================================================
# Factory
# ===================================================================


class TestBuildPerturbationSuite:
    def test_basic_config(self):
        config = {
            "action": [
                {"type": "ActionNoise", "std": [0.1, 0.2, 0.2, 0.2]},
                {"type": "ActionBias"},
            ],
            "observation_image": [
                {"type": "ImageNoise", "std": 15.0},
            ],
        }
        suite = build_perturbation_suite(config)
        assert len(suite.action) == 2
        assert len(suite.observation_image) == 1
        assert len(suite.environment_means) == 0

    def test_empty_config(self):
        suite = build_perturbation_suite({})
        assert len(suite.action) == 0

    def test_all_perturbation_types(self):
        config = {
            "action": [
                {"type": "ActionNoise"},
                {"type": "ActionBias"},
                {"type": "ActionScale"},
            ],
            "observation_image": [
                {"type": "ImageNoise"},
                {"type": "ImageOcclusion"},
                {"type": "BrightnessShift"},
            ],
            "observation_state": [
                {"type": "StateEstimateNoise"},
            ],
            "observation_camera": [
                {"type": "CameraPoseNoise"},
            ],
            "environment_means": [
                {"type": "GaussianShift"},
            ],
            "environment_scales": [
                {"type": "GaussianScale"},
            ],
            "environment_opacities": [
                {"type": "GaussianOpacity"},
            ],
        }
        suite = build_perturbation_suite(config)
        assert len(suite.action) == 3
        assert len(suite.observation_image) == 3
        assert len(suite.observation_state) == 1
        assert len(suite.observation_camera) == 1
        assert len(suite.environment_means) == 1
        assert len(suite.environment_scales) == 1
        assert len(suite.environment_opacities) == 1

    def test_unknown_type_raises(self):
        config = {"action": [{"type": "NonExistent"}]}
        with pytest.raises(KeyError):
            build_perturbation_suite(config)


# ===================================================================
# Quaternion helpers
# ===================================================================


class TestQuaternionMultiply:
    def test_identity(self):
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        result = _quat_multiply(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_self_inverse(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        result = _quat_multiply(q, q_conj)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestQuaternionMultiplyWxyzTorch:
    def test_identity(self):
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q = torch.tensor([[0.9, 0.1, 0.2, 0.3]])
        q = q / q.norm(dim=-1, keepdim=True)
        result = _quat_multiply_wxyz_torch(identity, q)
        torch.testing.assert_close(result, q)

    def test_batch(self):
        q1 = torch.randn(5, 4)
        q1 = q1 / q1.norm(dim=-1, keepdim=True)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(5, 4)
        result = _quat_multiply_wxyz_torch(identity, q1)
        torch.testing.assert_close(result, q1, atol=1e-6, rtol=1e-5)
