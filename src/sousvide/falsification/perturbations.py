"""
Perturbation engine for the falsification framework.

Three perturbation surfaces:
1. **Action perturbations** – modify the policy's output before it reaches the
   dynamics integrator.
2. **Observation perturbations** – modify images / state estimates before they
   reach the policy.
3. **Environment perturbations** – modify the Gaussian splat scene itself
   (shift/scale Gaussians, change opacities) so rendering produces different
   observations and the collision geometry changes.

Every perturbation is a stateful object that can be ``apply``-ed at each
timestep and ``reset`` between episodes.  Perturbations can be composed via
``PerturbationStack``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


# ===================================================================
# Base classes
# ===================================================================

class Perturbation(ABC):
    """Base class for all perturbations."""

    @abstractmethod
    def reset(self, rng: np.random.RandomState) -> None:
        """Re-sample any stochastic parameters for a new episode."""

    @abstractmethod
    def apply(self, **kwargs):
        """Apply the perturbation.  Subclass decides the signature."""


# ===================================================================
# 1.  ACTION perturbations
# ===================================================================

@dataclass
class ActionNoiseConfig:
    """Additive Gaussian noise on the control vector."""
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4))
    std: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.1, 0.1, 0.1]))


class ActionNoise(Perturbation):
    """Additive Gaussian noise on [thrust, wx, wy, wz]."""

    def __init__(self, config: ActionNoiseConfig | None = None):
        self.cfg = config or ActionNoiseConfig()
        self._rng: np.random.RandomState = np.random.RandomState()

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng

    def apply(self, action: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(loc=self.cfg.mean, scale=self.cfg.std)
        return action + noise


@dataclass
class ActionBiasConfig:
    """Constant bias added to the control vector (persists across an episode)."""
    bias_range: np.ndarray = field(
        default_factory=lambda: np.array([[-.05, .05], [-.2, .2], [-.2, .2], [-.2, .2]])
    )


class ActionBias(Perturbation):
    """Per-episode constant bias on [thrust, wx, wy, wz], sampled at reset."""

    def __init__(self, config: ActionBiasConfig | None = None):
        self.cfg = config or ActionBiasConfig()
        self._bias = np.zeros(4)

    def reset(self, rng: np.random.RandomState) -> None:
        lo, hi = self.cfg.bias_range[:, 0], self.cfg.bias_range[:, 1]
        self._bias = rng.uniform(lo, hi)

    def apply(self, action: np.ndarray) -> np.ndarray:
        return action + self._bias


@dataclass
class ActionScaleConfig:
    """Multiplicative scaling on the control vector."""
    scale_range: Tuple[float, float] = (0.8, 1.2)


class ActionScale(Perturbation):
    """Per-episode multiplicative scaling, sampled uniformly at reset."""

    def __init__(self, config: ActionScaleConfig | None = None):
        self.cfg = config or ActionScaleConfig()
        self._scale = 1.0

    def reset(self, rng: np.random.RandomState) -> None:
        self._scale = rng.uniform(*self.cfg.scale_range)

    def apply(self, action: np.ndarray) -> np.ndarray:
        return action * self._scale


# ===================================================================
# 2.  OBSERVATION perturbations
# ===================================================================

@dataclass
class ImageNoiseConfig:
    """Additive pixel-space Gaussian noise."""
    std: float = 10.0


class ImageNoise(Perturbation):
    """Gaussian noise injected into camera images before policy sees them."""

    def __init__(self, config: ImageNoiseConfig | None = None):
        self.cfg = config or ImageNoiseConfig()
        self._rng: np.random.RandomState = np.random.RandomState()

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng

    def apply(self, image: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0, self.cfg.std, size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@dataclass
class ImageOcclusionConfig:
    """Random rectangular occlusions (simulates partial sensor failure)."""
    num_patches: int = 2
    patch_size_range: Tuple[int, int] = (20, 60)
    color: Tuple[int, int, int] = (0, 0, 0)


class ImageOcclusion(Perturbation):
    """Drops random rectangular patches onto the image."""

    def __init__(self, config: ImageOcclusionConfig | None = None):
        self.cfg = config or ImageOcclusionConfig()
        self._rng: np.random.RandomState = np.random.RandomState()

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        for _ in range(self.cfg.num_patches):
            ps = self._rng.randint(*self.cfg.patch_size_range)
            x = self._rng.randint(0, max(w - ps, 1))
            y = self._rng.randint(0, max(h - ps, 1))
            img[y:y + ps, x:x + ps] = self.cfg.color
        return img


@dataclass
class BrightnessShiftConfig:
    """Global brightness shift (simulates lighting changes)."""
    shift_range: Tuple[float, float] = (-40.0, 40.0)


class BrightnessShift(Perturbation):
    """Per-episode constant brightness offset."""

    def __init__(self, config: BrightnessShiftConfig | None = None):
        self.cfg = config or BrightnessShiftConfig()
        self._shift = 0.0

    def reset(self, rng: np.random.RandomState) -> None:
        self._shift = rng.uniform(*self.cfg.shift_range)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.clip(image.astype(np.float32) + self._shift, 0, 255).astype(np.uint8)


@dataclass
class CameraPoseNoiseConfig:
    """Noise on the camera extrinsic used for rendering (perturbs what the
    policy *sees* without affecting true dynamics)."""
    position_std: float = 0.02          # metres
    rotation_std: float = 0.01          # radians (axis-angle magnitude)


class CameraPoseNoise(Perturbation):
    """Adds small perturbations to the camera-to-body transform before
    rendering, so the observation is misaligned with the true state."""

    def __init__(self, config: CameraPoseNoiseConfig | None = None):
        self.cfg = config or CameraPoseNoiseConfig()
        self._rng: np.random.RandomState = np.random.RandomState()

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng

    def apply(self, Tc2b: np.ndarray) -> np.ndarray:
        """Perturb a 4x4 camera-to-body transform."""
        T = Tc2b.copy()
        T[:3, 3] += self._rng.normal(0, self.cfg.position_std, 3)
        angle = self._rng.normal(0, self.cfg.rotation_std)
        axis = self._rng.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-8)
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        T[:3, :3] = R_noise @ T[:3, :3]
        return T


@dataclass
class StateEstimateNoiseConfig:
    """Noise on the state estimate passed to the policy."""
    position_std: float = 0.02
    velocity_std: float = 0.05
    orientation_std: float = 0.01


class StateEstimateNoise(Perturbation):
    """Perturbs the state vector [p, v, q] before the policy sees it."""

    def __init__(self, config: StateEstimateNoiseConfig | None = None):
        self.cfg = config or StateEstimateNoiseConfig()
        self._rng: np.random.RandomState = np.random.RandomState()

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng

    def apply(self, state: np.ndarray) -> np.ndarray:
        s = state.copy()
        s[:3] += self._rng.normal(0, self.cfg.position_std, 3)
        s[3:6] += self._rng.normal(0, self.cfg.velocity_std, 3)
        dq = self._rng.normal(0, self.cfg.orientation_std, 3)
        q = s[6:10]
        dq_full = np.array([*dq, 1.0])
        dq_full /= np.linalg.norm(dq_full)
        s[6:10] = _quat_multiply(q, dq_full)
        s[6:10] /= np.linalg.norm(s[6:10])
        return s


# ===================================================================
# 3.  ENVIRONMENT perturbations (Gaussian splat scene)
# ===================================================================

@dataclass
class GaussianShiftConfig:
    """Shift a subset of Gaussians to create / remove obstacles."""
    fraction: float = 0.05              # fraction of Gaussians affected
    shift_std: float = 0.05             # metres


class GaussianShift(Perturbation):
    """Translates a random subset of Gaussians in the scene.

    This physically alters the collision geometry, making some paths
    passable or impassable.
    """

    def __init__(self, config: GaussianShiftConfig | None = None):
        self.cfg = config or GaussianShiftConfig()
        self._indices: np.ndarray | None = None
        self._shifts: torch.Tensor | None = None
        self._original_means: torch.Tensor | None = None

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng
        self._indices = None
        self._shifts = None
        self._original_means = None

    def apply(self, gsplat_means: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        gsplat_means : (N, 3) tensor of Gaussian centres on the same device.

        Returns
        -------
        perturbed_means : (N, 3) tensor.
        """
        N = gsplat_means.shape[0]
        if self._indices is None:
            k = max(1, int(N * self.cfg.fraction))
            self._indices = self._rng.choice(N, size=k, replace=False)
            self._shifts = torch.from_numpy(
                self._rng.normal(0, self.cfg.shift_std, (k, 3)).astype(np.float32)
            ).to(gsplat_means.device)
            self._original_means = gsplat_means[self._indices].clone()

        out = gsplat_means.clone()
        out[self._indices] += self._shifts
        return out

    def restore(self, gsplat_means: torch.Tensor) -> torch.Tensor:
        """Undo the perturbation (call between episodes)."""
        if self._original_means is not None:
            out = gsplat_means.clone()
            out[self._indices] = self._original_means
            return out
        return gsplat_means


@dataclass
class GaussianScaleConfig:
    """Scale a subset of Gaussians (inflate/deflate obstacles)."""
    fraction: float = 0.05
    scale_range: Tuple[float, float] = (0.5, 2.0)


class GaussianScale(Perturbation):
    """Scales a random subset of Gaussians, making obstacles larger or
    smaller."""

    def __init__(self, config: GaussianScaleConfig | None = None):
        self.cfg = config or GaussianScaleConfig()
        self._indices: np.ndarray | None = None
        self._scales: torch.Tensor | None = None
        self._original_scales: torch.Tensor | None = None

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng
        self._indices = None
        self._scales = None
        self._original_scales = None

    def apply(self, gsplat_scales: torch.Tensor) -> torch.Tensor:
        N = gsplat_scales.shape[0]
        if self._indices is None:
            k = max(1, int(N * self.cfg.fraction))
            self._indices = self._rng.choice(N, size=k, replace=False)
            raw = self._rng.uniform(*self.cfg.scale_range, size=(k, 1)).astype(np.float32)
            self._scales = torch.from_numpy(raw).to(gsplat_scales.device)
            self._original_scales = gsplat_scales[self._indices].clone()

        out = gsplat_scales.clone()
        out[self._indices] *= self._scales
        return out

    def restore(self, gsplat_scales: torch.Tensor) -> torch.Tensor:
        if self._original_scales is not None:
            out = gsplat_scales.clone()
            out[self._indices] = self._original_scales
            return out
        return gsplat_scales


@dataclass
class GaussianOpacityConfig:
    """Modify opacities of Gaussians (make obstacles appear/disappear)."""
    fraction: float = 0.05
    opacity_range: Tuple[float, float] = (0.0, 1.0)


class GaussianOpacity(Perturbation):
    """Randomises opacity of a subset of Gaussians, causing obstacles to
    appear transparent or previously-transparent regions to become opaque."""

    def __init__(self, config: GaussianOpacityConfig | None = None):
        self.cfg = config or GaussianOpacityConfig()
        self._indices: np.ndarray | None = None
        self._opacities: torch.Tensor | None = None
        self._original_opacities: torch.Tensor | None = None

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng
        self._indices = None
        self._opacities = None
        self._original_opacities = None

    def apply(self, gsplat_opacities: torch.Tensor) -> torch.Tensor:
        N = gsplat_opacities.shape[0]
        if self._indices is None:
            k = max(1, int(N * self.cfg.fraction))
            self._indices = self._rng.choice(N, size=k, replace=False)
            raw = self._rng.uniform(*self.cfg.opacity_range, size=(k,)).astype(np.float32)
            self._opacities = torch.from_numpy(raw).to(gsplat_opacities.device)
            self._original_opacities = gsplat_opacities[self._indices].clone()

        out = gsplat_opacities.clone()
        out[self._indices] = self._opacities
        return out

    def restore(self, gsplat_opacities: torch.Tensor) -> torch.Tensor:
        if self._original_opacities is not None:
            out = gsplat_opacities.clone()
            out[self._indices] = self._original_opacities
            return out
        return gsplat_opacities


# ===================================================================
# Composition
# ===================================================================

class PerturbationStack:
    """Ordered collection of perturbations of the same surface (action / obs /
    env) that are applied sequentially."""

    def __init__(self, perturbations: Sequence[Perturbation] | None = None):
        self.perturbations: List[Perturbation] = list(perturbations or [])

    def reset(self, rng: np.random.RandomState) -> None:
        for p in self.perturbations:
            p.reset(rng)

    def apply(self, value, **kwargs):
        for p in self.perturbations:
            value = p.apply(value, **kwargs) if kwargs else p.apply(value)
        return value

    def __len__(self) -> int:
        return len(self.perturbations)


@dataclass
class PerturbationSuite:
    """Complete perturbation configuration for one falsification episode.

    Groups perturbations by surface so the orchestrator knows where to
    apply each one in the simulation loop.
    """
    action: PerturbationStack = field(default_factory=PerturbationStack)
    observation_image: PerturbationStack = field(default_factory=PerturbationStack)
    observation_state: PerturbationStack = field(default_factory=PerturbationStack)
    observation_camera: PerturbationStack = field(default_factory=PerturbationStack)
    environment_means: PerturbationStack = field(default_factory=PerturbationStack)
    environment_scales: PerturbationStack = field(default_factory=PerturbationStack)
    environment_opacities: PerturbationStack = field(default_factory=PerturbationStack)

    def reset_all(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        for stack in [self.action, self.observation_image,
                      self.observation_state, self.observation_camera,
                      self.environment_means, self.environment_scales,
                      self.environment_opacities]:
            stack.reset(rng)


# ===================================================================
# Helpers
# ===================================================================

def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


# ===================================================================
# Factory
# ===================================================================

def build_perturbation_suite(config: Dict) -> PerturbationSuite:
    """Build a PerturbationSuite from a plain dict (e.g. loaded from YAML).

    Expected schema::

        action:
          - type: ActionNoise
            std: [0.05, 0.1, 0.1, 0.1]
          - type: ActionBias
            bias_range: ...
        observation_image:
          - type: ImageNoise
            std: 15.0
        environment_means:
          - type: GaussianShift
            fraction: 0.03
            shift_std: 0.04
    """
    _registry: Dict[str, type] = {
        "ActionNoise": ActionNoise,
        "ActionBias": ActionBias,
        "ActionScale": ActionScale,
        "ImageNoise": ImageNoise,
        "ImageOcclusion": ImageOcclusion,
        "BrightnessShift": BrightnessShift,
        "CameraPoseNoise": CameraPoseNoise,
        "StateEstimateNoise": StateEstimateNoise,
        "GaussianShift": GaussianShift,
        "GaussianScale": GaussianScale,
        "GaussianOpacity": GaussianOpacity,
    }

    _config_registry: Dict[str, type] = {
        "ActionNoise": ActionNoiseConfig,
        "ActionBias": ActionBiasConfig,
        "ActionScale": ActionScaleConfig,
        "ImageNoise": ImageNoiseConfig,
        "ImageOcclusion": ImageOcclusionConfig,
        "BrightnessShift": BrightnessShiftConfig,
        "CameraPoseNoise": CameraPoseNoiseConfig,
        "StateEstimateNoise": StateEstimateNoiseConfig,
        "GaussianShift": GaussianShiftConfig,
        "GaussianScale": GaussianScaleConfig,
        "GaussianOpacity": GaussianOpacityConfig,
    }

    suite = PerturbationSuite()

    for surface_name in ["action", "observation_image", "observation_state",
                         "observation_camera", "environment_means",
                         "environment_scales", "environment_opacities"]:
        entries = config.get(surface_name, [])
        stack = getattr(suite, surface_name)
        for entry in entries:
            entry = dict(entry)
            cls_name = entry.pop("type")
            cls = _registry[cls_name]
            cfg_cls = _config_registry.get(cls_name)
            if cfg_cls and entry:
                cfg = cfg_cls(**entry)
                stack.perturbations.append(cls(cfg))
            else:
                stack.perturbations.append(cls())

    return suite
