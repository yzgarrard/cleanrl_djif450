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


class TacDroneHoverEnvV02(gym.Env):
    """
    ## Observation Space  (Box, shape=(21,), dtype=float32)
    Index | Quantity
    ------|----------------------------------------------------------
    0-2   | Position  error ex, ey, ez  (world frame)
    3-5   | Velocity vx, vy, vz  (world frame)
    6-8   | Acceleration ax, ay, az  (body frame, accel)
    9-12  | Quaternion  qw, qx, qy, qz
    13-15 | Angular velocity  wx, wy, wz  (body frame)
    16    | Height above the ground (z)
    17-20 | Previous action 

    ## Action Space  (Box, shape=(4,), dtype=float32)
    Normalised motor commands in [-1, 1] (re-scaled to [0, 13.5] N inside step).
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
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.frame_skip  = 1
        self.max_thrust  = 10

        # --- Spaces ---
        obs_low  = np.full(21, -np.inf, dtype=np.float32)
        obs_high = np.full(21,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Symmetric [-1, 1] → avoids SB3 warning; rescaled in step()
        self.action_space = spaces.Box(
            low  = -np.ones(4, dtype=np.float32),
            high =  np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        
        # Physical constants
        self.motor_time_constant = 0.059  # seconds, for first-order motor delay approximation
        self.alpha = np.exp(-0.01/self.motor_time_constant)  # assuming dt of 0.01 seconds

        # Desired pose
        self.target_pos = np.array([0,0,target_z], dtype=np.float32)
        
        #
        self.last_action = np.zeros(4, dtype=np.float32)
        self.ctrl_cmd = np.zeros(4, dtype=np.float32)
        
        # --- Reward weights ---
        self.w_pos = 3
        self.w_vel = 0.2
        self.w_att = 2
        self.w_omega = 0.1
        self.w_act = 0.05
        self.w_act_baseline = 1.883*9.81/4  # baseline force for hover thrust
        self.alive  = 2.0

        # --- Rendering (only used when render_mode="human" on eval env) ---
        self._viewer   = None
        self._viewer_launched = False
        self._renderer = None
        self._step_count = 0

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        pos   = self.data.qpos[:3].copy()
        pos_err = self.target_pos - pos
        quat  = self.data.qpos[3:7].copy()
        vel   = self.data.qvel[:3].copy()
        accel = self.data.sensor("body_accel").data.copy()
        gyro  = self.data.sensor("body_gyro").data.copy()
        return np.concatenate([pos_err, vel, accel, quat, gyro, np.array([pos[2]]), self.last_action]).astype(np.float32)

    def _tilt_angle(self) -> float:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        body_z   = rot.reshape(3, 3)[:, 2]
        cos_tilt = float(np.clip(body_z[2], -1.0, 1.0))
        return float(np.arccos(cos_tilt))

    def _compute_reward(self) -> float:
        pos  = self.data.qpos[:3]
        vel  = self.data.qvel[:3]
        gyro = self.data.sensor("body_gyro").data
        
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        
        pos_err = np.linalg.norm(self.target_pos - pos)**2
        vel_err = np.linalg.norm(vel)**2
        att_err = 1-self.data.qpos[3]**2
        omega_err = np.linalg.norm(gyro)**2
        act_cost = np.linalg.norm(self.ctrl_cmd - self.w_act_baseline)**2
        return  float(
              - self.w_pos  * pos_err
              - self.w_vel  * vel_err
              - self.w_att  * att_err
              - self.w_omega * omega_err
              - self.w_act  * act_cost
              + self.alive
        )

    def _is_terminated(self) -> bool:
        pos  = self.data.qpos[:3]
        tilt = self._tilt_angle()
        pos_xy_err = np.linalg.norm(self.target_pos[0:2] - pos[0:2])
        if pos_xy_err > 1 or tilt > np.deg2rad(60) or pos[2] < 0.05 or pos[2] > 3.0:
            return True
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
        self.data.qpos[2]  = 0.135 + rng.uniform(0, 0.01)
        
        self.last_action = np.zeros(4, dtype=np.float32)
        self.ctrl_cmd = np.zeros(4, dtype=np.float32)
        
        axis  = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-8
        angle = rng.uniform(0, np.deg2rad(5))
        q_pert = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_pert, axis, angle)
        q_out  = np.zeros(4)
        mujoco.mju_mulQuat(q_out, self.data.qpos[3:7].copy(), q_pert)
        self.data.qpos[3:7] = q_out

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Rescale [-1,1] → [0, max_thrust]
        self.ctrl_cmd = (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * self.max_thrust
        self.last_action = action.copy()
        # Set the thrust of the motor instantly
        # self.data.ctrl[:] = ctrl_cmd
        

        for _ in range(self.frame_skip):
            # Set the thrust of the motor accounting for motor delay
            self.data.ctrl[:] = self.alpha*self.data.ctrl[:] + (1-self.alpha)*self.ctrl_cmd
            mujoco.mj_step(self.model, self.data)


        obs        = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        if terminated: reward -= 100
        self._step_count += 1
        truncated  = self._step_count >= self.max_episode_steps

        info = {
            "z":        float(self.data.qpos[2]),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Rendering  (safe: only called on the dedicated eval/render env)    #
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "human":
            if not self._viewer_launched:
                import mujoco.viewer as mjv
                self._viewer = mjv.launch_passive(self.model, self.data)
                self._viewer_launched = True
            if self._viewer is not None and self._viewer.is_running():
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
        self._viewer_launched = False
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
