# DJI F450 Baseline Controller
import gymnasium as gym
import numpy as np
import custom_envs
import matplotlib.pyplot as plt

%matplotlib qt

seed = 0xf00e
environment_name = "custom_envs/DJIF450-v0"
env = gym.make(environment_name)
observation, info = env.reset(seed=seed)

sim_length_steps = 1000
observation_history = np.zeros((sim_length_steps, len(observation)), dtype=np.float32)


MPC_XYZ_POS_P = np.diag([0.95, 0.95, 1.0])
MPC_XYZ_VEL_P = np.diag([1.8, 1.8, 4.0])
MPC_XYZ_VEL_D = np.diag([0.2, 0.2, 0.0])

reward_total = 0

for i in range(sim_length_steps):
    pos_error = -observation[0:3]  # Desired position is (0, 0, 0)
    vel_sp = MPC_XYZ_POS_P @ pos_error
    vel_error = vel_sp - observation[3:6]
    acc_sp = MPC_XYZ_VEL_P @ vel_error - MPC_XYZ_VEL_D @ observation[3:6] 
    # action = np.array([0, 0, 0, 0], dtype=np.float32)  # No control input
    observation, reward, terminated, truncated, info = env.step(acc_sp)
    reward_total += reward
    observation_history[i] = observation
    
# env.close()

print(f"Cumulative Reward: {reward_total:.2f}")

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(observation_history[:, 0], label='x')
axs[0].plot(observation_history[:, 1], label='y')
axs[0].plot(observation_history[:, 2], label='z')
axs[0].set_title('Position')
axs[0].legend()
axs[0].grid("both")
axs[1].plot(observation_history[:, 3], label='vx')
axs[1].plot(observation_history[:, 4], label='vy')
axs[1].plot(observation_history[:, 5], label='vz')
axs[1].set_title('Velocity')
axs[1].legend()
axs[1].grid()
axs[2].plot(observation_history[:, 13], label='last_action_x')
axs[2].plot(observation_history[:, 14], label='last_action_y')
axs[2].plot(observation_history[:, 15], label='last_action_z')
axs[2].set_title('Last Action')
axs[2].legend()
axs[2].grid()
plt.tight_layout()
plt.show()