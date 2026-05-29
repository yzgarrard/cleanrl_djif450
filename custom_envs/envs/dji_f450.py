import numpy as np
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
from gymnasium import spaces

# This environment simulates the dynamics of a DJI F450 quadcopter drone in 3D space.
# The learning task is to learn the acceleration setpoints to drive the drone to the origin and hover there.

class DJIF450EnvV0p0(gym.Env):
    
    def __init__(self):
        
        ## Drone parameters
        self.m = 1.883  # kg
        self.g = 9.81  # m/s^2
        self.inertia = np.diag([
            0.01355, 
            0.01561, 
            0.02672])  # kg*m^2
        self.inertia_inv = np.diag(1/np.diag(self.inertia))
        self.arm_length = 0.225  # m
        # Simulation rate
        self.dt = 0.01  # s
        self.max_tilt = np.deg2rad(45)  # radians
        # Motor thrust coefficients: f_i = kf[0] + kf[1] * u_i + kf[2] * u_i^2, where u_i is in [0,1]
        # u here is normalized in [0,1]
        self.kf = np.array([4.6680, -9.0966, 17.6946])
        # Motor torque coefficient: \tau_i=cross(r_{p_i},r_{f_i})*f_i+r_{\tau_i}*K_{\tau_i}*f_i
        self.kt = 0.01671
        # Motor time constant: \dot{\omega} = (1/\tau) * (\omega_{sp} - \omega)
        self.tau = 0.059  # s
        # Initial motor rpm is hover
        self.motor_rpm = np.array([6500, 6500, 6500, 6500])
        # Initial collective thrust is hover thrust
        self.collective_thrust = -self.m * self.g
        # Initial torque is zero
        self.torque = np.array([0, 0, 0])
        # Mapping from normalized motor inputs to rpm
        self.usp_to_rpmsp = lambda u: -4825.1*u**2 + 12235*u + 1308
        # Update function for motor rpm
        alpha = np.exp(-self.dt/self.tau)
        self.update_motor_rpm = lambda rpm, rpm_sp: alpha * rpm + rpm_sp*(1 - alpha)
        self.update_collective_thrust = lambda collective_thrust, collective_thrust_sp: alpha * collective_thrust + collective_thrust_sp*(1 - alpha)
        self.update_torque = lambda torque, torque_sp: alpha * torque + torque_sp*(1 - alpha)
        # Position control gains
        self.MPC_XYZ_POS_P = np.diag([0.95, 0.95, 1.0])
        self.MPC_XYZ_VEL_P = np.diag([1.8, 1.8, 4.0])
        self.MPC_XYZ_VEL_D = np.diag([0.2, 0.2, 0.0])
        # Reward weights
        self.pos_reward_weight = 20
        self.vel_reward_weight = 0.5
        self.quat_reward_weight = 2.5
        self.omega_reward_weight = 0.0
        self.action_reward_weight = 0.05
        self.survival_reward = 2
        # 
        self.last_action = np.array([0, 0, 0], dtype=np.float32)
        
        ## Observation space: 13-dimensional space
        # Position (x, y, z) in NED frame
        # Velocity (vx, vy, vz) in NED frame
        # Quaternion (w, x, y, z) in world frame
        # Angular velocity (wx, wy, wz) in body frame
        self.observation_space = spaces.Box(
            low=np.array([-5,-5,-5,-5,-5,-5,-1,-1,-1,-1,-10,-10,-10,-10,-10,-10]), 
            high=np.array([5,5,5,5,5,5,1,1,1,1,10,10,10,10,10,10]), 
            shape=(16,), 
            dtype=np.float32)
        
        self._agent_pose = np.array(
            [0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0, 
             1.0, 0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0], 
            dtype=np.float32)
        
        ## Action space: 3-dimensional vector
        # Action: desired (ax, ay, az) acceleration in NED frame
        self.action_space = spaces.Box(
            low=np.array([-10,-10,-10]), 
            high=np.array([10,10,10]), 
            shape=(3,), 
            dtype=np.float32)
        
    def _get_obs(self):
        # Return the current observation
        return np.concatenate([self._agent_pose.copy(), self.last_action.copy()], dtype=np.float32)
    
    def _get_info(self):
        # Return additional information about the environment
        return {
            "state": self._agent_pose,
            "motor_rpm": self.motor_rpm
            }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ## Reset the environment to have some random pose
        self._agent_pose = np.array(
            [0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0, 
             1.0, 0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0], 
            dtype=np.float32)
        self._agent_pose[0:3] = self.np_random.uniform(-0.2, 0.2, 3)
        # self._agent_pose[0:3] = self.np_random.uniform(-1, 1, 3)
        # self._agent_pose[3:6] = self.np_random.uniform(-1, 1, 3)
        self._agent_pose[3:6] = np.array([0, 0, 0], dtype=np.float32)
        self.motor_rpm = np.array([6500, 6500, 6500, 6500], dtype=np.float32)
        self.collective_thrust = -self.m * self.g
        self.torque = np.array([0, 0, 0], dtype=np.float32)
        self.last_action = np.array([0, 0, 0], dtype=np.float32)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        pos = self._agent_pose[:3]
        vel = self._agent_pose[3:6]
        quat = self._agent_pose[6:10]
        rot = R.from_quat(quat)
        omega = self._agent_pose[10:13]


        ## Compute thrust setpoint
        # https://github.com/PX4/PX4-Autopilot/blob/e4d46f20f439094862eedd7e21c5abeefb1721f1/src/modules/mc_pos_control/PositionControl/PositionControl.cpp#L207
        z_specific_force = -self.g
        z_specific_force += action[2]
        # Assume standard acceleration due to gravity in the vertical direction for attitude generation,
        # i.e., the line below describes the orientation the drone needs to be in to generate thrust
        # in the action[0:1] direction.
        body_z = np.array([-action[0], -action[1], -z_specific_force])
        body_z = body_z / np.linalg.norm(body_z)

        # Limit the tilt angle
        # https://github.com/PX4/PX4-Autopilot/blob/eb9a76cfaff6357ff8b8845f256b99f7376607b4/src/modules/mc_pos_control/PositionControl/ControlMath.cpp#L53
        dot_product_unit = np.dot(body_z, np.array([0, 0, 1]))
        angle = np.arccos(dot_product_unit)
        angle = min(angle, self.max_tilt)
        rejection = body_z - (dot_product_unit * np.array([0, 0, 1]))
        if (np.linalg.norm(rejection)**2 < np.finfo(np.float64).eps):
            rejection[0] = 1
        body_z = np.cos(angle) * np.array([0, 0, 1]) + np.sin(angle) * rejection / np.linalg.norm(rejection)
        
        # Scale thrust assuming hover thrust produces standard gravity
        hover_thrust = self.m * self.g
        thrust_ned_z = action[2] * (hover_thrust / self.g) - hover_thrust   # should be negative in hover condition
        
        # Project thrust to planned body attitude
        cos_ned_body = np.dot(np.array([0,0,1]), body_z)
        collective_thrust = min(thrust_ned_z / cos_ned_body, -0.001)
        
        thrust_sp = body_z * collective_thrust
        
        ## Convert acceleration setpoint to attitude setpoint and thrust
        
        yaw_sp = 0
        
        # body_z = body_z / np.linalg.norm(body_z)
        body_z = -thrust_sp / np.linalg.norm(-thrust_sp)
        
        # vector of desired yaw direction in XY plane, rotate by PI/2
        y_C = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0])
        
        # desired body_x axis, orthogonal to body_z
        body_x = np.cross(y_C, body_z)
        
        # Keep nose to fron while inverted upside down
        if body_z[2] < 0:
            body_x = -body_x
            
        if (np.abs(body_z[2]) < np.finfo(np.float64).eps):
            # desired thrust is in XY plane, set X downside to construct correct
            # matrix, but yaw component will not be used actually
            body_x = np.array([0.0, 0.0, 1.0])
        
        body_x = body_x / np.linalg.norm(body_x)
        body_y = np.cross(body_z, body_x)
        dcm_rotm_sp = np.array([body_x, body_y, body_z]).transpose()
        quat_sp = R.from_matrix(dcm_rotm_sp).as_quat(scalar_first=True)
        
        ## Attitude controller from https://docs.px4.io/main/en/flight_stack/controller_diagrams
        Q_quat = lambda q: np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1],  q[0], -q[3],  q[2]],
            [q[2],  q[3],  q[0], -q[1]],
            [q[3], -q[2],  q[1],  q[0]]
        ])
        q_conj = lambda q: np.array([q[0], -q[1], -q[2], -q[3]])
        q_err = Q_quat(q_conj(quat)) @ quat_sp
        q_sign = np.sign(q_err[0])
        q_err_pregain = q_sign * q_err[1:4]
        omega_sp = np.diag([4, 4, 2.8]) @ q_err_pregain * 2
        
        omega_err = omega_sp - omega
        torque_sp = np.diag([0.15, 0.15, 0.2]) @ omega_err# - np.diag([0.003, 0.003, 0]) @ omega
        
        self.torque = self.update_torque(self.torque, torque_sp)
        self.collective_thrust = self.update_collective_thrust(self.collective_thrust, collective_thrust)
        # self.torque = torque_sp
        # self.collective_thrust = collective_thrust
        
        # Compute angular acceleration
        omega_dot = self.inertia_inv @ (self.torque + np.cross(self.inertia @ omega, omega))
        
        # Update the agent's state
        pos_dot = vel
        vel_dot = 1/self.m * R.from_quat(quat, scalar_first=True).as_matrix() @ np.array([0, 0, self.collective_thrust]) + np.array([0, 0, self.g])
        quat_dot = 0.5*np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ]) @ quat
        omega_dot = omega_dot
        
        self._agent_pose[0:3] += pos_dot * self.dt
        self._agent_pose[3:6] += vel_dot * self.dt
        self._agent_pose[6:10] += quat_dot * self.dt
        self._agent_pose[6:10] = self._agent_pose[6:10] / np.sqrt(np.sum(self._agent_pose[6:10]**2))    # Normalize the quaternion while updating the state
        self._agent_pose[10:13] += omega_dot * self.dt
        
        rotm = R.from_quat(self._agent_pose[6:10], scalar_first=True).as_matrix()
        rotm_desired = R.from_quat(quat_sp, scalar_first=True).as_matrix()
        
        reward = (
            -1*np.linalg.norm(pos) + 
            -(3-np.trace(rotm.T @ rotm_desired)) +
            -0.1*(np.linalg.norm(omega) + 2*np.linalg.norm(omega_dot)) +
            -0.01*(np.linalg.norm(action) + 2*np.linalg.norm(self.last_action - action)) +
            0.1
        )

        
        self.last_action = action
        terminated = False
        if np.linalg.norm(self._agent_pose[0:3]) > 2.25:
            terminated = True
            reward -= 100
        
        # return observation, reward, terminated truncated, info
        return self._get_obs(), reward, terminated, False, self._get_info()