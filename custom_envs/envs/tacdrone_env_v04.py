"""
TacDrone Hovering - Custom Gymnasium Environment
================================================
Fixed version: render_mode="human" is intentionally NOT used during
training to avoid Wayland/GLFW + DummyVecEnv segfault.
Rendering is handled externally via RenderCallback in the train script.

V4 uses thrust+angular rates rather than direct motor commands
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from scipy.spatial.transform import Rotation as R

_DEFAULT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tacdrone.xml")


class TacDroneHoverEnvV04(gym.Env):
    """
    ## Observation Space  (Box, shape=(20,), dtype=float32)
    Index | Quantity
    ------|----------------------------------------------------------
    0-2   | Position error ex, ey, ez  (world frame)
    3-6   | Quaternion  qw, qx, qy, qz
    7-9   | Linear velocity  vx, vy, vz  (world frame)
    10-12 | Angular velocity  wx, wy, wz  (body frame, gyro)
    13-15 | Linear acceleration ax, ay, az (body frame, accelerometer)
    16-19 | Previous action

    ## Action Space  (Box, shape=(4,), dtype=float32)
    Normalised thrust and angular rates in [-1, 1] (re-scaled to [0, 10.0] N and [rad/s] inside step).
    Angular rates are in xyz directions (or pqr, or roll/pitch/yaw rate) in the body frame.
    euler rates aren't actually euler rates, but are in body frame.
    Using symmetric [-1,1] per SB3 recommendation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
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
        self.max_thrust  = 40.0     # total thrust, not per motor
        self.dt = self.model.opt.timestep * self.frame_skip
        
        # Things for rate control
        self.MC_PITCHRATE_P = 0.15
        self.MC_PITCHRATE_I = 0.2
        self.MC_PITCHRATE_D = 0.003
        self.MC_ROLLRATE_P  = 0.15
        self.MC_ROLLRATE_I  = 0.2
        self.MC_ROLLRATE_D  = 0.003
        self.MC_YAWRATE_P   = 0.2
        self.MC_YAWRATE_I   = 0.1
        self.MC_YAWRATE_D   = 0.0
        self.pitchrate_err_accum = 0.0
        self.rollrate_err_accum  = 0.0
        self.yawrate_err_accum   = 0.0
        s45 = np.sin(np.deg2rad(45)) # arm angle
        d = 0.225   # Arm length
        K_tau = 0.0167  # Torque coefficient
        self.CAM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-d*s45, d*s45, d*s45, -d*s45],
            [-d*s45, d*s45, -d*s45, d*s45],
            [-K_tau, -K_tau, K_tau, K_tau]
            ])
        self.CAM_inv = np.linalg.inv(self.CAM)
        self.max_i_torque = np.array([0.03, 0.03, 0.01], dtype=np.float32) # anti-windup limits for the integral term of the rate controller

        # Motor stuff
        self.motor_time_constant = 0.059  # seconds, for first-order motor delay approximation
        self.alpha = np.exp(-self.dt/self.motor_time_constant)  # assuming dt of 0.01 seconds


        # --- Spaces ---
        obs_low  = np.full(20, -np.inf, dtype=np.float32)
        obs_high = np.full(20,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Symmetric [-1, 1] → avoids SB3 warning; rescaled in step()
        self.action_space = spaces.Box(
            low  = -np.ones(4, dtype=np.float32),
            high =  np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action = np.zeros(4, dtype=np.float32)
        hover_thrust = self.model.body_mass.sum() * 9.81
        self.hover_action = np.array(
            [2.0 * hover_thrust / self.max_thrust - 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        # --- Reward weights ---
        self.w_z    = 3.0
        self.w_xy   = 1.0
        self.w_pos_err = 3.0
        self.w_vel  = 0.2
        self.w_ang  = 0.1
        self.w_tilt = 2.0
        self.w_yaw = 1.0
        self.w_act  = 0.05
        self.w_act_delta = 0.1
        self.alive  = 0.0
        
        # Desired pos
        self.pos_des = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Last action
        self.last_action = np.zeros(4, dtype=np.float32)

        # --- Rendering (only used when render_mode="human" on eval env) ---
        self._viewer = None
        self._viewer_launched = False
        self._renderer = None
        self._step_count = 0
        self._init_xy = np.zeros(2)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        rng = self.np_random
        pos   = self.data.qpos[:3].copy() + rng.normal(0,0.0005, size=3)
        quat  = self.data.qpos[3:7].copy()  # Not sure how to add noise here
        vel   = self.data.qvel[:3].copy() + rng.normal(0,0.1, size=3) # Using https://docs.px4.io/main/en/advanced_config/parameter_reference#EKF2_EVV_NOISE
        gyro  = self.data.sensor("body_gyro").data.copy() + rng.normal(0,0.015, size=3) # Using https://docs.px4.io/main/en/advanced_config/parameter_reference#EKF2_GYR_NOISE
        accel = self.data.sensor("body_accel").data.copy() + rng.normal(0,0.35, size=3) # Using https://docs.px4.io/main/en/advanced_config/parameter_reference#EKF2_ACC_NOISE
        # z_err = np.array([self.pos_des - pos[2]], dtype=np.float32)
        # xy_err    = self.pos_des[0:2] - pos[:2]
        pos_err = (self.pos_des - pos).astype(np.float32)
        return np.concatenate([pos_err, quat, vel, gyro, accel, self.last_action]).astype(np.float32)

    def _tilt_angle(self) -> float:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        body_z   = rot.reshape(3, 3)[:, 2]
        cos_tilt = float(np.clip(body_z[2], -1.0, 1.0))
        return float(np.arccos(cos_tilt))

    def _compute_reward(self, action_normed: np.ndarray) -> tuple[float, dict[str, float]]:
        pos  = self.data.qpos[:3]
        pos_err = (self.pos_des - pos).astype(np.float32)
        vel  = self.data.qvel[:3]
        gyro = self.data.sensor("body_gyro").data
        z_err  = self.pos_des[2] - pos[2]
        xy_err = float(np.linalg.norm(self.pos_des[:2] - pos[:2]))
        tilt   = self._tilt_angle()
        quat = self.data.qpos[3:7]
        eul = R.from_quat(quat, scalar_first=True).as_euler("zyx", degrees=False)
        yaw = eul[0]
        action_delta = action_normed - self.last_action
        action_error = action_normed - self.hover_action
        reward_terms = {
            "alive": float(self.alive),
            "z": float(-self.w_z * z_err**2),
            "xy": float(-self.w_xy * xy_err**2),
            "vel": float(-self.w_vel * float(np.dot(vel, vel))),
            "ang": float(-self.w_ang * float(np.dot(gyro, gyro))),
            "tilt": float(-self.w_tilt * tilt**2),
            "yaw": float(-self.w_yaw * yaw**2),
            "act": float(-self.w_act * float(np.sum(action_error**2))),
            "act_delta": float(-self.w_act_delta * float(np.sum(action_delta**2))),
            "termination": 0.0,
        }
        reward_terms["total"] = float(sum(reward_terms.values()))
        return reward_terms["total"], reward_terms

    def _termination_reason(self) -> str | None:
        pos  = self.data.qpos[:3]
        tilt = self._tilt_angle()
        if pos[2] < 0.05:
            return "ground_contact"
        if abs(self.pos_des[2] - pos[2]) > 3.0:
            return "altitude_error"
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0:
            return "xy_bounds"
        if tilt > np.deg2rad(60):
            return "excessive_tilt"
        return None

    def _is_terminated(self) -> bool:
        return self._termination_reason() is not None

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        self.data.qpos[0] = rng.uniform(-0.5, 0.5)
        self.data.qpos[1] = rng.uniform(-0.5, 0.5)
        self.data.qpos[2]  = 0.135 + rng.uniform(0, 1)
        
        self.pitchrate_err_accum = 0.0
        self.rollrate_err_accum  = 0.0
        self.yawrate_err_accum   = 0.0
        
        self.pos_des[0:2] = rng.uniform(-0.5, 0.5, size=2)
        self.pos_des[2] = rng.uniform(0.5, 2.0)
        
        self.last_action = np.zeros(4, dtype=np.float32)

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
        thrust_sp = (np.clip(action[0], -1.0, 1.0) + 1.0) * 0.5 * self.max_thrust
        omega_sp = np.clip(action[1:], -1.0, 1.0) # Keep body rates within 1 rad/s
        omega = self.data.sensor("body_gyro").data.copy()
        omega_err = omega_sp - omega
        
        self.rollrate_err_accum += omega_err[0]*self.dt
        self.pitchrate_err_accum += omega_err[1]*self.dt
        self.yawrate_err_accum   += omega_err[2]*self.dt

        i_gains = np.array([
            self.MC_ROLLRATE_I,
            self.MC_PITCHRATE_I,
            self.MC_YAWRATE_I,
        ])
        i_limits = np.divide(
            self.max_i_torque,
            i_gains,
            out=np.zeros_like(self.max_i_torque),
            where=i_gains != 0.0,
        )
        self.rollrate_err_accum = np.clip(
            self.rollrate_err_accum, -i_limits[0], i_limits[0]
        )
        self.pitchrate_err_accum = np.clip(
            self.pitchrate_err_accum, -i_limits[1], i_limits[1]
        )
        self.yawrate_err_accum = np.clip(
            self.yawrate_err_accum, -i_limits[2], i_limits[2]
        )

        
        tau_sp_P = np.array([
            self.MC_ROLLRATE_P * omega_err[0],
            self.MC_PITCHRATE_P * omega_err[1],
            self.MC_YAWRATE_P   * omega_err[2]
        ])
        tau_sp_I = np.array([
            self.MC_ROLLRATE_I * self.rollrate_err_accum,
            self.MC_PITCHRATE_I * self.pitchrate_err_accum,
            self.MC_YAWRATE_I   * self.yawrate_err_accum
        ])
        tau_sp_D = -np.array([
            self.MC_ROLLRATE_D * omega[0],
            self.MC_PITCHRATE_D * omega[1],
            self.MC_YAWRATE_D   * omega[2]
        ])
        tau_sp = tau_sp_P + tau_sp_I + tau_sp_D
        motor_force_cmd = self.CAM_inv @ np.concatenate([ [thrust_sp], tau_sp ])
        motor_force_cmd = np.clip(motor_force_cmd, 0.0, self.max_thrust/4)

        for _ in range(self.frame_skip):
            # self.data.ctrl[:] = motor_force_cmd
            self.data.ctrl[:] = self.alpha*self.data.ctrl[:] + (1-self.alpha)*motor_force_cmd
            mujoco.mj_step(self.model, self.data)
            
        obs        = self._get_obs()
        reward, reward_terms = self._compute_reward(action)
        termination_reason = self._termination_reason()
        terminated = termination_reason is not None
        if terminated:
            reward_terms["termination"] = -200.0  # large penalty for crashing/going out of bounds
            reward = float(reward + reward_terms["termination"])
            reward_terms["total"] = reward
        self._step_count += 1
        truncated  = self._step_count >= self.max_episode_steps
        if truncated and termination_reason is None:
            termination_reason = "time_limit"
        self.last_action = action.copy()

        info = {
            "z":        float(self.data.qpos[2]),
            "z_err":    float(self.pos_des[2] - self.data.qpos[2]),
            "tilt_deg": float(np.rad2deg(self._tilt_angle())),
            "x_err": float(self.pos_des[0] - self.data.qpos[0]),
            "y_err": float(self.pos_des[1] - self.data.qpos[1]),
            "pos_des": self.pos_des.copy(),
            "reward_terms": reward_terms,
            "termination_reason": termination_reason,
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
