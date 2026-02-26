"""
Failure detection and safety tracking for the falsification framework.

Monitors the drone state during VLA rollout and determines:
1. Whether the current state is "safe"
2. Whether a failure has occurred
3. What the last safe state was before failure

Safety / failure criteria are **pluggable** – users register ``SafetyCriterion``
objects that independently vote on whether the current state is safe.  The
detector fuses these votes (all must pass for "safe").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===================================================================
# Trajectory snapshot
# ===================================================================

@dataclass
class StateSnapshot:
    """Complete snapshot of the sim state at one control timestep."""
    time: float
    state: np.ndarray                   # [px,py,pz, vx,vy,vz, qx,qy,qz,qw]
    control: np.ndarray                 # [thrust, wx, wy, wz]
    rgb_forward: np.ndarray | None = None
    rgb_downward: np.ndarray | None = None
    depth: np.ndarray | None = None
    step_index: int = 0


class FailureType(Enum):
    NONE = auto()
    COLLISION = auto()
    OUT_OF_BOUNDS = auto()
    EXCESSIVE_VELOCITY = auto()
    EXCESSIVE_TILT = auto()
    GOAL_DIVERGENCE = auto()
    CUSTOM = auto()


@dataclass
class FailureRecord:
    """Detailed record of a detected failure."""
    failure_type: FailureType
    description: str
    failure_step: int
    failure_state: StateSnapshot
    last_safe_step: int
    last_safe_state: StateSnapshot
    trajectory_up_to_failure: List[StateSnapshot]
    metadata: Dict = field(default_factory=dict)


# ===================================================================
# Safety criteria (pluggable)
# ===================================================================

class SafetyCriterion(ABC):
    """A single test that returns True when the state is safe."""

    @abstractmethod
    def is_safe(self, snapshot: StateSnapshot) -> Tuple[bool, str]:
        """
        Returns
        -------
        safe : bool
        reason : str   (empty when safe, explains violation otherwise)
        """

    @abstractmethod
    def failure_type(self) -> FailureType:
        """The category of failure this criterion detects."""


class BoundsCriterion(SafetyCriterion):
    """Fails if position exits an axis-aligned bounding box."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def is_safe(self, snapshot: StateSnapshot) -> Tuple[bool, str]:
        pos = snapshot.state[:3]
        if np.any(pos < self.lower) or np.any(pos > self.upper):
            return False, (
                f"Position {pos} outside bounds [{self.lower}, {self.upper}]"
            )
        return True, ""

    def failure_type(self) -> FailureType:
        return FailureType.OUT_OF_BOUNDS


class VelocityCriterion(SafetyCriterion):
    """Fails if speed exceeds a threshold."""

    def __init__(self, max_speed: float = 5.0):
        self.max_speed = max_speed

    def is_safe(self, snapshot: StateSnapshot) -> Tuple[bool, str]:
        speed = np.linalg.norm(snapshot.state[3:6])
        if speed > self.max_speed:
            return False, f"Speed {speed:.2f} exceeds {self.max_speed:.2f} m/s"
        return True, ""

    def failure_type(self) -> FailureType:
        return FailureType.EXCESSIVE_VELOCITY


class TiltCriterion(SafetyCriterion):
    """Fails if the tilt angle from vertical exceeds a threshold.

    Uses the quaternion [qx,qy,qz,qw] at state indices [6:10].
    In NED convention, "upright" means the body z-axis is aligned with
    world +z (i.e. pointing down).  We compute the angle between the
    body z-axis and the world z-axis.
    """

    def __init__(self, max_tilt_deg: float = 60.0):
        self.max_tilt_rad = np.radians(max_tilt_deg)

    def is_safe(self, snapshot: StateSnapshot) -> Tuple[bool, str]:
        q = snapshot.state[6:10]
        qx, qy, qz, qw = q
        body_z = np.array([
            2 * (qx * qz + qy * qw),
            2 * (qy * qz - qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ])
        cos_angle = np.clip(np.dot(body_z, np.array([0, 0, 1])), -1, 1)
        tilt = np.arccos(cos_angle)
        if tilt > self.max_tilt_rad:
            return False, (
                f"Tilt {np.degrees(tilt):.1f} deg exceeds "
                f"{np.degrees(self.max_tilt_rad):.1f} deg"
            )
        return True, ""

    def failure_type(self) -> FailureType:
        return FailureType.EXCESSIVE_TILT


class GoalDivergenceCriterion(SafetyCriterion):
    """Fails if the drone moves too far from the goal *after* it should be
    converging (i.e. after a configurable fraction of the episode)."""

    def __init__(self, goal_position: np.ndarray, max_distance: float = 2.0,
                 active_after_fraction: float = 0.5):
        self.goal = np.asarray(goal_position, dtype=float)
        self.max_distance = max_distance
        self.active_after_fraction = active_after_fraction
        self._total_steps: int = 0

    def set_total_steps(self, n: int) -> None:
        self._total_steps = n

    def is_safe(self, snapshot: StateSnapshot) -> Tuple[bool, str]:
        if self._total_steps > 0:
            frac = snapshot.step_index / self._total_steps
            if frac < self.active_after_fraction:
                return True, ""
        dist = np.linalg.norm(snapshot.state[:3] - self.goal)
        if dist > self.max_distance:
            return False, (
                f"Distance to goal {dist:.2f}m exceeds {self.max_distance:.2f}m"
            )
        return True, ""

    def failure_type(self) -> FailureType:
        return FailureType.GOAL_DIVERGENCE


class ProximityCollisionCriterion(SafetyCriterion):
    """Heuristic collision check: uses a set of known obstacle positions and
    radii.  For full Gaussian-based collision checking, use the SplatNav
    collision set instead (see ``splatnav_recovery.py``)."""

    def __init__(self, obstacles: List[Tuple[np.ndarray, float]] | None = None,
                 robot_radius: float = 0.05):
        self.obstacles = obstacles or []
        self.robot_radius = robot_radius

    def is_safe(self, snapshot: StateSnapshot) -> Tuple[bool, str]:
        pos = snapshot.state[:3]
        for center, radius in self.obstacles:
            dist = np.linalg.norm(pos - np.asarray(center))
            if dist < radius + self.robot_radius:
                return False, (
                    f"Collision: distance {dist:.3f}m to obstacle at "
                    f"{center} (threshold {radius + self.robot_radius:.3f}m)"
                )
        return True, ""

    def failure_type(self) -> FailureType:
        return FailureType.COLLISION


# ===================================================================
# Main detector
# ===================================================================

class FailureDetector:
    """
    Monitors a VLA rollout and detects failures in real time.

    Usage::

        detector = FailureDetector(criteria=[...])
        for step in rollout:
            snap = StateSnapshot(...)
            failed, record = detector.step(snap)
            if failed:
                # record.last_safe_state has the recovery start point
                break
    """

    def __init__(
        self,
        criteria: Sequence[SafetyCriterion] | None = None,
        safe_horizon: int = 3,
    ):
        """
        Parameters
        ----------
        criteria : list of SafetyCriterion
        safe_horizon : int
            Number of consecutive safe steps required before a state is
            considered truly "safe" (filters transient spikes).
        """
        self.criteria = list(criteria or [])
        self.safe_horizon = safe_horizon

        self._history: List[StateSnapshot] = []
        self._safe_streak: int = 0
        self._last_safe_idx: int = 0
        self._last_safe_snap: StateSnapshot | None = None
        self._failed: bool = False

    def reset(self) -> None:
        self._history.clear()
        self._safe_streak = 0
        self._last_safe_idx = 0
        self._last_safe_snap = None
        self._failed = False

    def step(self, snapshot: StateSnapshot) -> Tuple[bool, Optional[FailureRecord]]:
        """Process one timestep.

        Returns
        -------
        failed : bool
        record : FailureRecord or None (only set when failed is True)
        """
        self._history.append(snapshot)

        for criterion in self.criteria:
            safe, reason = criterion.is_safe(snapshot)
            if not safe:
                self._failed = True
                last_safe = self._last_safe_snap or self._history[0]
                record = FailureRecord(
                    failure_type=criterion.failure_type(),
                    description=reason,
                    failure_step=snapshot.step_index,
                    failure_state=snapshot,
                    last_safe_step=self._last_safe_idx,
                    last_safe_state=last_safe,
                    trajectory_up_to_failure=list(self._history),
                )
                return True, record

        # All criteria passed — update safe streak
        self._safe_streak += 1
        if self._safe_streak >= self.safe_horizon:
            self._last_safe_idx = snapshot.step_index
            self._last_safe_snap = snapshot

        return False, None

    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def last_safe_state(self) -> Optional[StateSnapshot]:
        return self._last_safe_snap

    @property
    def trajectory(self) -> List[StateSnapshot]:
        return list(self._history)
