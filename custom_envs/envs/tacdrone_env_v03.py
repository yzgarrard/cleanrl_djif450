"""
TacDrone Hovering - Custom Gymnasium Environment
================================================
Fixed version: render_mode="human" is intentionally NOT used during
training to avoid Wayland/GLFW + DummyVecEnv segfault.
Rendering is handled externally via RenderCallback in the train script.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_DEFAULT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tacdrone.xml")


class TacDroneHoverEnvV03(gym.Env):
    """
    ## Observation Space  (Box, shape=(19,), dtype=float32)
    Index | Quantity
    ------|----------------------------------------------------------
    0-2   | Position  x, y, z  (world frame)
    3-6   | Quaternion  qw, qx, qy, qz
    7-9   | Linear velocity  vx, vy, vz  (world frame)
    10-12 | Angular velocity  wx, wy, wz  (body frame, gyro)
    13-15 | Linear acceleration ax, ay, az  (body frame, accel)
    16    | z-error  (target_z - current_z)
    17-18 | xy displacement from spawn  (dx, dy)

    ## Action Space  (Box, shape=(4,), dtype=float32)
    Normalised motor commands in [-1, 1] (re-scaled to [0, 10.0] N inside step).
    Using symmetric [-1,1] per SB3 recommendation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        target_z: float = 1.0,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()

        self.xml_path = xml_path
        self.target_z = target_z
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.frame_skip  = 1
        self.max_thrust  = 10.0

        # --- Spaces ---
        obs_low  = np.full(19, -np.inf, dtype=np.float32)
        obs_high = np.full(19,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Symmetric [-1, 1] → avoids SB3 warning; rescaled in step()
        self.action_space = spaces.Box(
            low  = -np.ones(4, dtype=np.float32),
            high =  np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # --- Reward weights ---
        self.w_z    = 3.0
        self.w_xy   = 1.0
        self.w_vel  = 0.2
        self.w_ang  = 0.1
        self.w_tilt = 2.0
        self.w_act  = 0.05
        self.alive  = 1.0

        # --- Rendering (only used when render_mode="human" on eval env) ---
        self._viewer   = None
        self._renderer = None
        self._step_count = 0
        self._init_xy    = np.zeros(2)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        pos   = self.data.qpos[:3].copy()
        quat  = self.data.qpos[3:7].copy()
        vel   = self.data.qvel[:3].copy()
        gyro  = self.data.sensor("body_gyro").data.copy()
        accel = self.data.sensor("body_accel").data.copy()
        z_err = np.array([self.target_z - pos[2]], dtype=np.float32)
        xy    = pos[:2] - self._init_xy
        return np.concatenate([pos, quat, vel, gyro, accel, z_err, xy]).astype(np.float32)

    def _tilt_angle(self) -> float:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        body_z   = rot.reshape(3, 3)[:, 2]
        cos_tilt = float(np.clip(body_z[2], -1.0, 1.0))
        return float(np.arccos(cos_tilt))

    def _compute_reward(self, action_normed: np.ndarray) -> float:
        pos  = self.data.qpos[:3]
        vel  = self.data.qvel[:3]
        gyro = self.data.sensor("body_gyro").data
        z_err  = self.target_z - pos[2]
        xy_err = float(np.linalg.norm(pos[:2] - self._init_xy))
        tilt   = self._tilt_angle()
        return float(
              self.alive
            - self.w_z    * z_err**2
            - self.w_xy   * xy_err**2
            - self.w_vel  * float(np.dot(vel, vel))
            - self.w_ang  * float(np.dot(gyro, gyro))
            - self.w_tilt * tilt**2
            - self.w_act  * float(np.sum(action_normed**2))
        )

    def _is_terminated(self) -> bool:
        pos  = self.data.qpos[:3]
        tilt = self._tilt_angle()
        if pos[2] < 0.05:                            return True
        if abs(self.target_z - pos[2]) > 3.0:        return True
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0:  return True
        if tilt > np.deg2rad(60):                     return True
        return False

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        self.data.qpos[0] += rng.uniform(-0.05, 0.05)
        self.data.qpos[1] += rng.uniform(-0.05, 0.05)
        self.data.qpos[2]  = 0.135 + rng.uniform(-0.02, 0.02)

        axis  = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-8
        angle = rng.uniform(0, np.deg2rad(5))
        q_pert = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_pert, axis, angle)
        q_out  = np.zeros(4)
        mujoco.mju_mulQuat(q_out, self.data.qpos[3:7].copy(), q_pert)
        self.data.qpos[3:7] = q_out

        mujoco.mj_forward(self.model, self.data)
        self._init_xy    = self.data.qpos[:2].copy()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Rescale [-1,1] → [0, max_thrust]
        ctrl_cmd = (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * self.max_thrust
        self.data.ctrl[:] = ctrl_cmd

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated()
        self._step_count += 1
        truncated  = self._step_count >= self.max_episode_steps

        info = {
            "z":        float(self.data.qpos[2]),
            "z_err":    float(self.target_z - self.data.qpos[2]),
            "tilt_deg": float(np.rad2deg(self._tilt_angle())),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Rendering  (safe: only called on the dedicated eval/render env)    #
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mjv
                self._viewer = mjv.launch_passive(self.model, self.data)
            if self._viewer.is_running():
                self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
