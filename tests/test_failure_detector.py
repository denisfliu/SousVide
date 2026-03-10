"""Tests for the failure detection module (vla_falsification.falsification.failure_detector)."""

import numpy as np
import pytest

from vla_falsification.falsification.failure_detector import (
    BoundsCriterion,
    FailureDetector,
    FailureRecord,
    FailureType,
    GoalDivergenceCriterion,
    ProximityCollisionCriterion,
    StateSnapshot,
    TiltCriterion,
    VelocityCriterion,
)


# ===================================================================
# Helpers
# ===================================================================


def make_snapshot(
    pos=(0, 0, 0),
    vel=(0, 0, 0),
    quat=(0, 0, 0, 1),
    step=0,
    time=0.0,
):
    state = np.concatenate([
        np.array(pos, dtype=float),
        np.array(vel, dtype=float),
        np.array(quat, dtype=float),
    ])
    return StateSnapshot(
        time=time,
        state=state,
        control=np.zeros(4),
        step_index=step,
    )


# ===================================================================
# BoundsCriterion
# ===================================================================


class TestBoundsCriterion:
    def test_inside_bounds(self):
        c = BoundsCriterion(lower=[-1, -1, -1], upper=[1, 1, 1])
        snap = make_snapshot(pos=(0, 0, 0))
        safe, reason = c.is_safe(snap)
        assert safe
        assert reason == ""

    def test_on_boundary(self):
        c = BoundsCriterion(lower=[-1, -1, -1], upper=[1, 1, 1])
        snap = make_snapshot(pos=(1, 1, 1))
        safe, _ = c.is_safe(snap)
        assert safe

    def test_outside_bounds(self):
        c = BoundsCriterion(lower=[-1, -1, -1], upper=[1, 1, 1])
        snap = make_snapshot(pos=(2, 0, 0))
        safe, reason = c.is_safe(snap)
        assert not safe
        assert "outside bounds" in reason.lower() or "Position" in reason

    def test_failure_type(self):
        c = BoundsCriterion(lower=[0, 0, 0], upper=[1, 1, 1])
        assert c.failure_type() == FailureType.OUT_OF_BOUNDS


# ===================================================================
# VelocityCriterion
# ===================================================================


class TestVelocityCriterion:
    def test_below_threshold(self):
        c = VelocityCriterion(max_speed=5.0)
        snap = make_snapshot(vel=(1, 1, 1))
        safe, _ = c.is_safe(snap)
        assert safe

    def test_above_threshold(self):
        c = VelocityCriterion(max_speed=2.0)
        snap = make_snapshot(vel=(2, 2, 2))  # speed ~3.46
        safe, reason = c.is_safe(snap)
        assert not safe
        assert "speed" in reason.lower() or "Speed" in reason

    def test_exact_threshold(self):
        c = VelocityCriterion(max_speed=5.0)
        snap = make_snapshot(vel=(3, 4, 0))  # speed = 5.0
        safe, _ = c.is_safe(snap)
        assert safe

    def test_failure_type(self):
        c = VelocityCriterion()
        assert c.failure_type() == FailureType.EXCESSIVE_VELOCITY


# ===================================================================
# TiltCriterion
# ===================================================================


class TestTiltCriterion:
    def test_upright(self):
        c = TiltCriterion(max_tilt_deg=30.0)
        snap = make_snapshot()  # identity quaternion → 0 tilt
        safe, _ = c.is_safe(snap)
        assert safe

    def test_inverted(self):
        # 180° rotation around x-axis: quat = [1, 0, 0, 0]
        c = TiltCriterion(max_tilt_deg=90.0)
        snap = make_snapshot(quat=(1, 0, 0, 0))
        safe, reason = c.is_safe(snap)
        assert not safe

    def test_moderate_tilt(self):
        # 45° rotation around x-axis
        angle = np.radians(45)
        quat = np.array([np.sin(angle / 2), 0, 0, np.cos(angle / 2)])
        c = TiltCriterion(max_tilt_deg=60.0)
        snap = make_snapshot(quat=quat)
        safe, _ = c.is_safe(snap)
        assert safe

    def test_failure_type(self):
        c = TiltCriterion()
        assert c.failure_type() == FailureType.EXCESSIVE_TILT


# ===================================================================
# GoalDivergenceCriterion
# ===================================================================


class TestGoalDivergenceCriterion:
    def test_early_steps_always_safe(self):
        c = GoalDivergenceCriterion(
            goal_position=np.array([0, 0, 0]),
            max_distance=1.0,
            active_after_fraction=0.5,
        )
        c.set_total_steps(100)
        snap = make_snapshot(pos=(5, 5, 5), step=10)
        safe, _ = c.is_safe(snap)
        assert safe

    def test_late_steps_fail(self):
        c = GoalDivergenceCriterion(
            goal_position=np.array([0, 0, 0]),
            max_distance=1.0,
            active_after_fraction=0.5,
        )
        c.set_total_steps(100)
        snap = make_snapshot(pos=(5, 5, 5), step=80)
        safe, _ = c.is_safe(snap)
        assert not safe

    def test_close_to_goal(self):
        c = GoalDivergenceCriterion(
            goal_position=np.array([1, 0, 0]),
            max_distance=2.0,
            active_after_fraction=0.5,
        )
        c.set_total_steps(100)
        snap = make_snapshot(pos=(0.5, 0, 0), step=80)
        safe, _ = c.is_safe(snap)
        assert safe

    def test_failure_type(self):
        c = GoalDivergenceCriterion(goal_position=np.zeros(3))
        assert c.failure_type() == FailureType.GOAL_DIVERGENCE


# ===================================================================
# ProximityCollisionCriterion
# ===================================================================


class TestProximityCollisionCriterion:
    def test_no_obstacles(self):
        c = ProximityCollisionCriterion()
        snap = make_snapshot(pos=(0, 0, 0))
        safe, _ = c.is_safe(snap)
        assert safe

    def test_collision(self):
        obstacles = [(np.array([1, 0, 0]), 0.5)]
        c = ProximityCollisionCriterion(obstacles=obstacles, robot_radius=0.1)
        snap = make_snapshot(pos=(0.8, 0, 0))  # dist=0.2, threshold=0.6
        safe, reason = c.is_safe(snap)
        assert not safe
        assert "collision" in reason.lower() or "Collision" in reason

    def test_just_outside(self):
        obstacles = [(np.array([1, 0, 0]), 0.3)]
        c = ProximityCollisionCriterion(obstacles=obstacles, robot_radius=0.1)
        snap = make_snapshot(pos=(0.5, 0, 0))  # dist=0.5, threshold=0.4
        safe, _ = c.is_safe(snap)
        assert safe

    def test_failure_type(self):
        c = ProximityCollisionCriterion()
        assert c.failure_type() == FailureType.COLLISION


# ===================================================================
# FailureDetector
# ===================================================================


class TestFailureDetector:
    def test_no_criteria_always_safe(self):
        det = FailureDetector(criteria=[])
        for i in range(10):
            snap = make_snapshot(step=i)
            failed, record = det.step(snap)
            assert not failed
            assert record is None

    def test_detects_out_of_bounds(self):
        det = FailureDetector(
            criteria=[BoundsCriterion([-1, -1, -1], [1, 1, 1])],
            safe_horizon=1,
        )
        # A few safe steps
        for i in range(5):
            failed, _ = det.step(make_snapshot(pos=(0, 0, 0), step=i))
            assert not failed

        # Failure
        failed, record = det.step(make_snapshot(pos=(2, 0, 0), step=5))
        assert failed
        assert record is not None
        assert record.failure_type == FailureType.OUT_OF_BOUNDS
        assert record.failure_step == 5

    def test_safe_horizon(self):
        det = FailureDetector(
            criteria=[BoundsCriterion([-1, -1, -1], [1, 1, 1])],
            safe_horizon=3,
        )
        # 2 safe steps — not enough for safe_horizon=3
        det.step(make_snapshot(pos=(0, 0, 0), step=0))
        det.step(make_snapshot(pos=(0, 0, 0), step=1))
        assert det.last_safe_state is None  # not yet

        # 3rd safe step → now we have a safe state
        det.step(make_snapshot(pos=(0, 0, 0), step=2))
        assert det.last_safe_state is not None
        assert det.last_safe_state.step_index == 2

    def test_failure_record_has_trajectory(self):
        det = FailureDetector(
            criteria=[VelocityCriterion(max_speed=1.0)],
            safe_horizon=1,
        )
        det.step(make_snapshot(vel=(0, 0, 0), step=0))
        det.step(make_snapshot(vel=(0.5, 0, 0), step=1))
        failed, record = det.step(make_snapshot(vel=(2, 0, 0), step=2))
        assert failed
        assert len(record.trajectory_up_to_failure) == 3
        assert record.last_safe_step == 1

    def test_reset(self):
        det = FailureDetector(criteria=[VelocityCriterion(max_speed=1.0)])
        det.step(make_snapshot(vel=(2, 0, 0), step=0))
        assert det.is_failed

        det.reset()
        assert not det.is_failed
        assert det.last_safe_state is None
        assert len(det.trajectory) == 0

    def test_multiple_criteria(self):
        det = FailureDetector(criteria=[
            BoundsCriterion([-1, -1, -1], [1, 1, 1]),
            VelocityCriterion(max_speed=3.0),
        ])
        # Speed violation triggers first
        failed, record = det.step(make_snapshot(vel=(4, 0, 0), step=0))
        assert failed
        assert record.failure_type == FailureType.EXCESSIVE_VELOCITY


# ===================================================================
# StateSnapshot
# ===================================================================


class TestStateSnapshot:
    def test_fields(self):
        s = StateSnapshot(
            time=1.5,
            state=np.zeros(10),
            control=np.zeros(4),
            step_index=3,
        )
        assert s.time == 1.5
        assert s.step_index == 3
        assert s.rgb_forward is None
        assert s.depth is None
