"""
VLA (Vision-Language-Action) policy wrapper for FiGS.

Connects to an OpenPI server over websocket, sends dual-camera observations
+ a text prompt, receives a *chunk* of position-delta actions (typically 50),
and converts them into waypoints tracked by VehicleRateMPC.

Architecture
------------
1. The orchestrator renders cameras and calls ``query_actions()`` to get a
   full chunk of position deltas from the VLA server.
2. For each action in the chunk, the orchestrator calls ``step_action()``
   which accumulates the remaining deltas into waypoints and runs MPC to
   produce [thrust, wx, wy, wz].
3. After all actions in the chunk are consumed, the orchestrator renders
   new images and queries the server again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from figs.control.base_controller import BaseController


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VLAPolicyConfig:
    """Configuration for the VLA policy wrapper."""
    # Server
    host: str = "moraband"
    port: int = 8000

    # Task prompt
    prompt: str = "fly through the gate and hover over the stuffed penguin"

    # Control
    hz: int = 10
    actions_per_chunk: int = 50

    # Observation
    image_size: int = 256
    mask_third_person: bool = True

    # MPC tracking
    frame: str = "carl"

    # Coordinate conversion: VLA trained in MOCAP (Z-up), sim runs in NED.
    # Permutation ID for NED <-> MOCAP sign flips (default 5 = diag([1,-1,-1])).
    permutation: int = 5


# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------

#: Host alias registry. Call ``register_policy_host()`` to add entries.
POLICY_HOSTS: dict[str, str] = {
    "moraband": "moraband.stanford.edu",
    "manaan": "SOE-50TJK74.stanford.edu",
    "coruscant": "coruscant.stanford.edu",
}


def register_policy_host(alias: str, fqdn: str) -> None:
    """Register a host alias for the VLA policy server."""
    POLICY_HOSTS[alias] = fqdn


def _resolve_host(host: str) -> str:
    return POLICY_HOSTS.get(host, host)


# ---------------------------------------------------------------------------
# Build tXUd from waypoints (avoids MinTimeSnap rebuild)
# ---------------------------------------------------------------------------

def _waypoints_to_tXUd(
    current_state: np.ndarray,
    waypoints_ned: np.ndarray,
    hz: int,
    m: float,
    kt: float,
    n_rotors: int,
    g: float = 9.81,
) -> np.ndarray:
    """Build a desired trajectory array (tXUd) from waypoints.

    tXUd shape: (N, 15) — [t, px,py,pz, vx,vy,vz, qx,qy,qz,qw, T,wx,wy,wz]

    We linearly interpolate between current position and waypoints,
    compute velocities via finite differences, and set hover controls.
    """
    dt = 1.0 / hz
    hover_thrust = -(m * g) / (n_rotors * kt)

    pos_current = current_state[:3]
    vel_current = current_state[3:6] if len(current_state) >= 6 else np.zeros(3)
    quat_current = current_state[6:10] if len(current_state) >= 10 else np.array([0, 0, 0, 1.0])

    # Densify: insert intermediate points between waypoints at hz rate
    # Each waypoint is 1 control step apart (dt = 1/hz)
    all_positions = np.vstack([pos_current[np.newaxis, :], waypoints_ned])
    N = len(all_positions)

    # Pad to have enough horizon for the MPC (at least ~20 points)
    min_pts = 30
    if N < min_pts:
        pad = np.tile(all_positions[-1:], (min_pts - N, 1))
        all_positions = np.vstack([all_positions, pad])
        N = len(all_positions)

    tXUd = np.zeros((N, 15))
    for i in range(N):
        tXUd[i, 0] = i * dt                # time
        tXUd[i, 1:4] = all_positions[i]    # position

        # Velocity via finite difference
        if i == 0:
            tXUd[i, 4:7] = vel_current
        elif i < len(waypoints_ned) + 1:
            tXUd[i, 4:7] = (all_positions[i] - all_positions[i - 1]) / dt
        else:
            tXUd[i, 4:7] = 0.0  # padded points have zero velocity

        # Quaternion (identity / current)
        tXUd[i, 7:11] = quat_current

        # Controls (hover)
        tXUd[i, 11] = hover_thrust
        tXUd[i, 12:15] = 0.0

    return tXUd


# ---------------------------------------------------------------------------
# Main policy wrapper
# ---------------------------------------------------------------------------

class VLAPolicy(BaseController):
    """
    FiGS-compatible controller backed by an OpenPI VLA server.

    The orchestrator drives the control loop:
    1. Render cameras, call ``query_actions()`` → get a full action chunk.
    2. For each action in the chunk, call ``step_action()`` → MPC control.
    3. After the chunk is consumed, render again and repeat.
    """

    def __init__(self, config: VLAPolicyConfig | None = None):
        super().__init__()

        self.config = config or VLAPolicyConfig()

        self.name = "VLAPolicy"
        self.hz = self.config.hz

        # Dual-camera state
        self._rgb_downward: np.ndarray | None = None
        self._last_time: float = 0.0

        # OpenPI client (lazy-connect on first inference)
        self._client = None
        self._connected = False

        # MPC controller (built lazily on first control call)
        self._mpc = None
        self._mpc_built = False

        # Drone params (filled on first MPC build)
        self._m = None
        self._kt = None
        self._n_rotors = None
        self._g = None

        # Frame spec cache (set by orchestrator or resolved on demand)
        self._frame_spec = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the websocket connection to the VLA server."""
        if self._client is not None and hasattr(self._client, '_ws'):
            try:
                self._client._ws.close()
            except Exception:
                pass
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return

        from openpi_client import websocket_client_policy

        resolved = _resolve_host(self.config.host)
        print(f"[VLAPolicy] Connecting to OpenPI server at {resolved}:{self.config.port} ...", flush=True)
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=resolved, port=self.config.port,
        )
        self._connected = True
        print(f"[VLAPolicy] Connected.", flush=True)

    # ------------------------------------------------------------------
    # MPC builder (called once)
    # ------------------------------------------------------------------

    def build_mpc(self, x0: np.ndarray, frame_spec: dict = None) -> None:
        """Build the VehicleRateMPC once from a hover course at x0.

        Parameters
        ----------
        frame_spec : dict, optional
            Pre-resolved frame specification dict (from ``qs.generate_specifications``).
            If None, resolves from ``self.config.frame`` via config_helper.
        """
        if self._mpc_built:
            return

        from figs.control.vehicle_rate_mpc import VehicleRateMPC

        if frame_spec is not None:
            self._m = frame_spec["m"]
            self._kt = frame_spec["kt"]
            self._n_rotors = frame_spec["Nrtr"]
            self._g = frame_spec["g"]
        else:
            import figs.utilities.config_helper as ch
            import figs.dynamics.quadcopter_specifications as qs
            frame_dict = ch.get_config(self.config.frame, "frames")
            spec = qs.generate_specifications(frame_dict)
            self._m = spec["m"]
            self._kt = spec["kt"]
            self._n_rotors = spec["Nrtr"]
            self._g = spec["g"]

        pos = x0[:3]
        hover_course = {
            "waypoints": {
                "Nco": 6,
                "keyframes": {
                    "start": {
                        "t": 0.0,
                        "fo": [
                            [float(pos[0]), 0.0],
                            [float(pos[1]), 0.0],
                            [float(pos[2]), 0.0],
                            [0.0],
                        ],
                    },
                    "end": {
                        "t": 5.0,
                        "fo": [
                            [float(pos[0]), 0.0],
                            [float(pos[1]), 0.0],
                            [float(pos[2]), 0.0],
                            [0.0],
                        ],
                    },
                },
            },
            "forces": None,
        }

        custom_policy = {
            "plan": {"kT": 10, "use_l2_time": False},
            "track": {
                "hz": self.hz,
                "horizon": 10,
                "Qk": [100, 100, 100, 1, 1, 1, 10, 10, 10, 10],
                "Rk": [1, 1, 1, 1],
                "QN": [100, 100, 100, 1, 1, 1, 10, 10, 10, 10],
                "Ws": [10, 10, 10, 0.1, 0.1, 0.1, 0, 0, 0, 0],
                "bounds": {
                    "lower": [-1.0, -10.0, -10.0, -10.0],
                    "upper": [1.0, 10.0, 10.0, 10.0],
                },
            },
        }

        print("[VLAPolicy] Building MPC (one-time ACADOS setup)...", flush=True)
        t0 = time.time()
        self._mpc = VehicleRateMPC(
            policy=custom_policy,
            course=hover_course,
            frame=self.config.frame,
        )
        t1 = time.time()
        print(f"[VLAPolicy] MPC built in {t1-t0:.2f}s", flush=True)
        self._mpc_built = True

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_obs(
        self,
        rgb_forward: np.ndarray,
        rgb_downward: np.ndarray,
        state: np.ndarray,
    ) -> dict:
        """Build the observation dict expected by the OpenPI server.

        The VLA was trained in MOCAP (Z-up) frame, so we convert the NED
        state position to MOCAP before sending.
        """
        from vla_falsification.utilities.coordinate_transform import _get_perm_diag

        sz = self.config.image_size

        front = _resize(rgb_forward, sz)
        down = _resize(rgb_downward, sz)
        third = np.zeros((sz, sz, 3), dtype=np.uint8)

        # Convert NED position to MOCAP for the VLA
        P = _get_perm_diag(self.config.permutation)
        pos_mocap = P @ state[:3]

        state_vec = np.zeros(7, dtype=np.float32)
        state_vec[:3] = pos_mocap
        if len(state) >= 10:
            qx, qy, qz, qw = state[6:10]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy**2 + qz**2))
            state_vec[3] = yaw

        return {
            "observation/image": front,
            "observation/wrist_image": down,
            "observation/state": state_vec,
            "observation/front_1": front,
            "observation/down_1": down,
            "observation/3pov_1": third,
            "prompt": self.config.prompt,
        }

    # ------------------------------------------------------------------
    # Server inference
    # ------------------------------------------------------------------

    def query_actions(
        self,
        rgb_forward: np.ndarray,
        rgb_downward: np.ndarray,
        state: np.ndarray,
    ) -> List[np.ndarray]:
        """Query the VLA server and return the full action chunk.

        Returns
        -------
        actions : list of np.ndarray
            Each element is one action (position delta) from the chunk.
        """
        self._ensure_connected()

        obs = self.build_obs(rgb_forward, rgb_downward, state)

        print(f"      [VLA] querying server...", end="", flush=True)
        t0 = time.time()
        result = self._client.infer(obs)
        t1 = time.time()
        print(f" done ({t1-t0:.2f}s)", flush=True)
        raw = result["actions"]

        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            actions = [raw[i] for i in range(raw.shape[0])]
        elif isinstance(raw, (list, tuple)):
            actions = [np.asarray(a) for a in raw]
        else:
            actions = [np.asarray(raw)]

        return actions

    # ------------------------------------------------------------------
    # Single-action MPC step
    # ------------------------------------------------------------------

    def step_action(
        self,
        xcr: np.ndarray,
        upr: np.ndarray,
        action_index: int,
        actions: List[np.ndarray],
        rgb: np.ndarray = None,
        dpt: np.ndarray = None,
        fcr: np.ndarray = None,
    ) -> Tuple[np.ndarray, dict]:
        """Execute one action from the chunk via MPC tracking.

        Parameters
        ----------
        xcr : current state (10-d NED)
        upr : previous control
        action_index : index of the current action in ``actions``
        actions : the full action chunk from ``query_actions()``
        rgb, dpt, fcr : passed through to MPC (unused by MPC but required by interface)
        """
        if not self._mpc_built:
            self.build_mpc(xcr, frame_spec=self._frame_spec)

        from vla_falsification.utilities.coordinate_transform import _get_perm_diag

        vla_action = actions[action_index]
        current_pos = xcr[:3].copy()

        # VLA actions are position deltas in MOCAP frame; convert to NED.
        # P is its own inverse: P @ P = I
        P = _get_perm_diag(self.config.permutation)

        # Build waypoints: current action + remaining actions as lookahead
        remaining = actions[action_index:]
        lookahead = min(len(remaining), 21)  # current + up to 20 ahead

        waypoints = []
        wp = current_pos.copy()
        for j in range(lookahead):
            delta_ned = P @ remaining[j][:3]
            wp = wp + delta_ned
            waypoints.append(wp.copy())

        waypoints_arr = np.array(waypoints)

        new_tXUd = _waypoints_to_tXUd(
            current_state=xcr,
            waypoints_ned=waypoints_arr,
            hz=self.hz,
            m=self._m,
            kt=self._kt,
            n_rotors=self._n_rotors,
            g=self._g,
        )
        self._mpc.tXUd = new_tXUd

        _rgb = rgb if rgb is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        _dpt = dpt if dpt is not None else np.zeros((1, 1), dtype=np.float32)
        _fcr = fcr if fcr is not None else np.zeros(6)

        ucr, tsol_mpc = self._mpc.control(0.0, xcr, upr, _rgb, _dpt, _fcr)

        tsol = {
            "raw_vla_action": vla_action.copy(),
            "next_waypoint_ned": waypoints[0].copy(),
            "n_lookahead": len(waypoints),
            "action_index": action_index,
            "chunk_size": len(actions),
        }
        return ucr, tsol

    # ------------------------------------------------------------------
    # Legacy BaseController interface (kept for compatibility)
    # ------------------------------------------------------------------

    def control(
        self,
        tcr: float,
        xcr: np.ndarray,
        upr: np.ndarray,
        rgb: np.ndarray,
        dpt: np.ndarray,
        fcr: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        raise NotImplementedError(
            "VLAPolicy.control() is no longer used. "
            "Use query_actions() + step_action() from the orchestrator loop."
        )

    def reset_memory(
        self,
        x0: np.ndarray,
        u0: np.ndarray = None,
        fts0=None,
        pch0=None,
    ) -> None:
        self._rgb_downward = None
        self._last_time = 0.0
        if self._client is not None:
            try:
                self._client.reset()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if img.shape[0] != size or img.shape[1] != size:
        return cv2.resize(img, (size, size))
    return img
