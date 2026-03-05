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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


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


@dataclass
class GateRigidTransformConfig:
    """Apply a bounded rigid transform to gate Gaussians.

    This perturbation is strict by default: if it cannot sample a valid
    non-trivial transform that satisfies table-clearance constraints, it raises.
    """

    gate_mask_path: str = ""
    gate_points_path: str = ""
    table_points_path: str = ""
    max_match_distance_m: float = 0.01
    max_translation_m: Tuple[float, float, float] = (0.05, 0.05, 0.03)
    yaw_range_deg: Tuple[float, float] = (-8.0, 8.0)
    min_translation_m: float = 0.002
    min_abs_yaw_deg: float = 0.5
    min_table_clearance_m: float = 0.03
    max_sampling_tries: int = 100
    strict: bool = True


class GateRigidTransform(Perturbation):
    """Perturb only gate Gaussians with bounded translation + yaw."""

    def __init__(self, config: GateRigidTransformConfig | None = None):
        self.cfg = config or GateRigidTransformConfig()
        self._rng: np.random.RandomState = np.random.RandomState()
        self._mask: np.ndarray | None = None
        self._gate_points_ref: np.ndarray | None = None
        self._gate_indices: np.ndarray | None = None
        self._table_points: np.ndarray | None = None
        self._translation: np.ndarray | None = None
        self._yaw_rad: float | None = None
        self._rotation_np: np.ndarray | None = None
        self._sampled: bool = False

    def reset(self, rng: np.random.RandomState) -> None:
        self._rng = rng
        self._translation = None
        self._yaw_rad = None
        self._rotation_np = None
        self._sampled = False
        self._gate_indices = None
        if self._mask is None:
            if self.cfg.gate_mask_path:
                mask_path = Path(self.cfg.gate_mask_path)
                if not mask_path.exists():
                    raise FileNotFoundError(f"Gate mask not found: {mask_path}")
                mask = np.asarray(np.load(mask_path), dtype=bool).reshape(-1)
                if mask.ndim != 1 or mask.size == 0:
                    raise ValueError(f"Invalid gate mask shape from: {mask_path}")
                self._mask = mask
            elif not self.cfg.gate_points_path:
                raise ValueError(
                    "GateRigidTransform requires gate_mask_path and/or gate_points_path."
                )
        if self._gate_points_ref is None and self.cfg.gate_points_path:
            gate_points_path = Path(self.cfg.gate_points_path)
            if not gate_points_path.exists():
                raise FileNotFoundError(f"Gate points not found: {gate_points_path}")
            pts = np.asarray(np.load(gate_points_path), dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                raise ValueError(f"Invalid gate points shape from: {gate_points_path}")
            self._gate_points_ref = pts
        if self._table_points is None:
            if not self.cfg.table_points_path:
                raise ValueError("GateRigidTransform requires table_points_path.")
            table_path = Path(self.cfg.table_points_path)
            if not table_path.exists():
                raise FileNotFoundError(f"Table points not found: {table_path}")
            pts = np.asarray(np.load(table_path), dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                raise ValueError(f"Invalid table points shape from: {table_path}")
            self._table_points = pts

    def _resolve_gate_indices(self, gsplat_means: torch.Tensor) -> None:
        """Resolve gate indices on the current model means."""
        n_means = gsplat_means.shape[0]

        # Prefer coordinate-based matching when gate points are available.
        if self._gate_points_ref is None:
            if self._mask is not None and self._mask.shape[0] == n_means:
                self._gate_indices = np.where(self._mask)[0]
                if self._gate_indices.size == 0:
                    raise ValueError("Gate mask produced no selected gaussians.")
                return
            raise ValueError("Could not resolve gate indices: no valid gate points or mask.")

        means_np = gsplat_means.detach().cpu().numpy()
        mean_tree = cKDTree(means_np)
        dists, idx = mean_tree.query(self._gate_points_ref, k=1)
        max_dist = float(np.max(dists))
        if max_dist > self.cfg.max_match_distance_m:
            raise ValueError(
                "Gate point to model-mean matching too far: "
                f"max distance {max_dist:.6f}m exceeds {self.cfg.max_match_distance_m:.6f}m"
            )
        unique_idx = np.unique(idx.astype(np.int64))
        if unique_idx.size == 0:
            raise ValueError("Gate point matching resulted in zero gaussian indices.")

        # Optional sanity check against mask if mask aligns with current scene.
        if self._mask is not None and self._mask.shape[0] == n_means:
            mask_idx = np.where(self._mask)[0]
            overlap = np.intersect1d(unique_idx, mask_idx).size
            if overlap == 0:
                raise ValueError(
                    "Gate point matching found no overlap with provided gate mask; "
                    "artifacts likely come from different scenes."
                )
        self._gate_indices = unique_idx

    def _sample_valid_transform(self, gate_points: np.ndarray) -> None:
        assert self._table_points is not None
        table_tree = cKDTree(self._table_points)
        center = gate_points.mean(axis=0, keepdims=True)
        max_tx, max_ty, max_tz = self.cfg.max_translation_m
        yaw_min, yaw_max = self.cfg.yaw_range_deg
        last_failure = "unknown"

        for _ in range(self.cfg.max_sampling_tries):
            t = np.array(
                [
                    self._rng.uniform(-max_tx, max_tx),
                    self._rng.uniform(-max_ty, max_ty),
                    self._rng.uniform(-max_tz, max_tz),
                ],
                dtype=np.float64,
            )
            yaw_deg = float(self._rng.uniform(yaw_min, yaw_max))

            if (
                np.linalg.norm(t) < self.cfg.min_translation_m
                and abs(yaw_deg) < self.cfg.min_abs_yaw_deg
            ):
                last_failure = "transform too close to identity"
                continue

            yaw_rad = np.deg2rad(yaw_deg)
            cos_yaw = float(np.cos(yaw_rad))
            sin_yaw = float(np.sin(yaw_rad))
            rot = np.array(
                [
                    [cos_yaw, -sin_yaw, 0.0],
                    [sin_yaw, cos_yaw, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            transformed = (gate_points - center) @ rot.T + center + t
            min_dist = float(table_tree.query(transformed, k=1)[0].min())
            if min_dist < self.cfg.min_table_clearance_m:
                last_failure = (
                    f"table clearance violation ({min_dist:.4f}m < "
                    f"{self.cfg.min_table_clearance_m:.4f}m)"
                )
                continue

            self._translation = t
            self._yaw_rad = yaw_rad
            self._rotation_np = rot
            self._sampled = True
            return

        msg = (
            "GateRigidTransform failed to sample a valid non-identity transform "
            f"in {self.cfg.max_sampling_tries} tries; last failure: {last_failure}"
        )
        if self.cfg.strict:
            raise RuntimeError(msg)
        raise RuntimeError(msg)

    def apply(self, gsplat_means: torch.Tensor) -> torch.Tensor:
        if self._table_points is None:
            raise RuntimeError("GateRigidTransform used before reset().")
        if gsplat_means.ndim != 2 or gsplat_means.shape[1] != 3:
            raise ValueError("Expected gsplat_means shape (N, 3).")
        if self._gate_indices is None:
            self._resolve_gate_indices(gsplat_means)

        gate_points = gsplat_means[self._gate_indices].detach().cpu().numpy()
        if not self._sampled:
            self._sample_valid_transform(gate_points)
        assert self._translation is not None and self._rotation_np is not None

        center = gate_points.mean(axis=0, keepdims=True)
        transformed_gate = (gate_points - center) @ self._rotation_np.T + center + self._translation

        out = gsplat_means.clone()
        transformed_gate_t = torch.from_numpy(transformed_gate).to(
            device=gsplat_means.device, dtype=gsplat_means.dtype
        )
        out[self._gate_indices] = transformed_gate_t
        return out

    def apply_quats(self, gsplat_quats: torch.Tensor) -> torch.Tensor:
        """Rotate gate Gaussian orientations by sampled yaw (wxyz convention)."""
        if self._gate_indices is None or self._yaw_rad is None:
            raise RuntimeError("GateRigidTransform quaternion update called before sampling.")
        out = gsplat_quats.clone()
        q_delta = torch.tensor(
            [np.cos(self._yaw_rad / 2.0), 0.0, 0.0, np.sin(self._yaw_rad / 2.0)],
            device=gsplat_quats.device,
            dtype=gsplat_quats.dtype,
        )
        gate_quats = out[self._gate_indices]
        out[self._gate_indices] = _quat_multiply_wxyz_torch(q_delta[None, :], gate_quats)
        out[self._gate_indices] = torch.nn.functional.normalize(out[self._gate_indices], dim=-1)
        return out


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


def _quat_multiply_wxyz_torch(
    q1: torch.Tensor, q2: torch.Tensor
) -> torch.Tensor:
    """Hamilton product for quaternions in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


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
        "GateRigidTransform": GateRigidTransform,
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
        "GateRigidTransform": GateRigidTransformConfig,
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
