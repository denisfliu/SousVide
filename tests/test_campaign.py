"""Tests for campaign summarization and recovery utilities."""

import numpy as np
import pytest

from sousvide.falsification.failure_detector import (
    FailureRecord,
    FailureType,
    StateSnapshot,
)
from sousvide.falsification.orchestrator import (
    FalsificationEpisode,
    summarize_campaign,
)
from sousvide.falsification.splatnav_recovery import (
    RecoveryConfig,
    RecoveryResult,
    SplatNavRecovery,
)


# ===================================================================
# Helpers
# ===================================================================


def make_episode(
    episode_id=0,
    seed=0,
    success=True,
    failure_type=None,
    failure_step=0,
    recovery_feasible=None,
    wall_time=1.0,
    n_steps=10,
):
    trajectory = [
        StateSnapshot(
            time=float(i),
            state=np.zeros(10),
            control=np.zeros(4),
            step_index=i,
        )
        for i in range(n_steps)
    ]

    failure_record = None
    if failure_type is not None:
        failure_record = FailureRecord(
            failure_type=failure_type,
            description="test failure",
            failure_step=failure_step,
            failure_state=trajectory[-1],
            last_safe_step=max(0, failure_step - 1),
            last_safe_state=trajectory[0],
            trajectory_up_to_failure=trajectory,
        )

    recovery_result = None
    if recovery_feasible is not None:
        recovery_result = RecoveryResult(
            feasible=recovery_feasible,
            safe_state=trajectory[0],
            goal_position=np.zeros(3),
        )

    return FalsificationEpisode(
        episode_id=episode_id,
        seed=seed,
        success=success,
        trajectory=trajectory,
        failure_record=failure_record,
        recovery_result=recovery_result,
        wall_time_s=wall_time,
    )


# ===================================================================
# summarize_campaign
# ===================================================================


class TestSummarizeCampaign:
    def test_all_success(self):
        episodes = [make_episode(i, success=True) for i in range(5)]
        summary = summarize_campaign(episodes)
        assert summary["total_episodes"] == 5
        assert summary["successes"] == 5
        assert summary["failures"] == 0
        assert summary["failure_rate"] == 0.0
        assert summary["avg_failure_step"] is None

    def test_all_failures(self):
        episodes = [
            make_episode(
                i, success=False,
                failure_type=FailureType.OUT_OF_BOUNDS,
                failure_step=5 + i,
            )
            for i in range(3)
        ]
        summary = summarize_campaign(episodes)
        assert summary["total_episodes"] == 3
        assert summary["failures"] == 3
        assert summary["failure_rate"] == 1.0
        assert summary["avg_failure_step"] == 6.0  # mean(5,6,7)
        assert summary["failure_types"]["OUT_OF_BOUNDS"] == 3

    def test_mixed_results(self):
        episodes = [
            make_episode(0, success=True),
            make_episode(1, success=False, failure_type=FailureType.COLLISION, failure_step=3),
            make_episode(2, success=False, failure_type=FailureType.EXCESSIVE_VELOCITY,
                         failure_step=7, recovery_feasible=True),
        ]
        summary = summarize_campaign(episodes)
        assert summary["total_episodes"] == 3
        assert summary["successes"] == 1
        assert summary["failures"] == 2
        assert summary["recovered"] == 1
        assert summary["failure_types"]["COLLISION"] == 1
        assert summary["failure_types"]["EXCESSIVE_VELOCITY"] == 1

    def test_empty_campaign(self):
        summary = summarize_campaign([])
        assert summary["total_episodes"] == 0
        assert summary["failure_rate"] == 0

    def test_recovery_rate(self):
        episodes = [
            make_episode(0, success=False, failure_type=FailureType.COLLISION,
                         failure_step=5, recovery_feasible=True),
            make_episode(1, success=False, failure_type=FailureType.COLLISION,
                         failure_step=5, recovery_feasible=False),
            make_episode(2, success=False, failure_type=FailureType.COLLISION,
                         failure_step=5, recovery_feasible=True),
        ]
        summary = summarize_campaign(episodes)
        assert summary["recovered"] == 2
        assert summary["recovery_rate"] == pytest.approx(2 / 3)


# ===================================================================
# Recovery static helpers
# ===================================================================


class TestDownsampleTrajectory:
    def test_already_short(self):
        pos = np.random.randn(10, 3)
        result = SplatNavRecovery.downsample_trajectory(pos, max_waypoints=20)
        np.testing.assert_array_equal(result, pos)

    def test_downsamples(self):
        pos = np.random.randn(100, 3)
        result = SplatNavRecovery.downsample_trajectory(pos, max_waypoints=10)
        assert len(result) == 10
        # First and last preserved
        np.testing.assert_array_equal(result[0], pos[0])
        np.testing.assert_array_equal(result[-1], pos[-1])

    def test_exact_max(self):
        pos = np.random.randn(20, 3)
        result = SplatNavRecovery.downsample_trajectory(pos, max_waypoints=20)
        np.testing.assert_array_equal(result, pos)


class TestTrajectoryToFigsWaypoints:
    def test_feasible_trajectory(self):
        snap = StateSnapshot(time=0, state=np.zeros(10), control=np.zeros(4))
        result = RecoveryResult(
            feasible=True,
            safe_state=snap,
            goal_position=np.array([1, 0, 0]),
            trajectory_positions=np.array([
                [0, 0, 0],
                [0.5, 0, 0],
                [1, 0, 0],
            ]),
        )
        waypoints = SplatNavRecovery.trajectory_to_figs_waypoints(result, total_time=3.0)
        assert waypoints is not None
        assert "waypoints" in waypoints
        kf = waypoints["waypoints"]["keyframes"]
        assert len(kf) == 3
        # First waypoint at t=0
        assert list(kf.values())[0]["t"] == 0.0
        # Last waypoint at t=3
        assert list(kf.values())[-1]["t"] == 3.0

    def test_infeasible_returns_none(self):
        snap = StateSnapshot(time=0, state=np.zeros(10), control=np.zeros(4))
        result = RecoveryResult(
            feasible=False,
            safe_state=snap,
            goal_position=np.zeros(3),
        )
        assert SplatNavRecovery.trajectory_to_figs_waypoints(result) is None


# ===================================================================
# RecoveryConfig
# ===================================================================


class TestRecoveryConfig:
    def test_defaults(self):
        cfg = RecoveryConfig()
        assert cfg.robot_radius == 0.02
        assert cfg.vmax == 2.0
        assert cfg.spline_degree == 6

    def test_custom(self):
        cfg = RecoveryConfig(robot_radius=0.05, vmax=3.0)
        assert cfg.robot_radius == 0.05
        assert cfg.vmax == 3.0
