# DJI F450 Baseline Controller
import gymnasium as gym
import numpy as np
import custom_envs
import matplotlib.pyplot as plt
import torch

from cleanrl_drone.deploy_policy import DronePolicy

%matplotlib qt

seed = 0xf00e
environment_name = "custom_envs/DJIF450-v1"
env = gym.make(environment_name)
observation, info = env.reset(seed=seed)

sim_length_steps = 1000
observation_history = np.zeros((sim_length_steps, len(observation)), dtype=np.float32)
agent = torch.load(
    "/home/yzgar/cleanrl/runs/custom_envs/DJIF450-v0__ppo_continuous_action_v01__4__20260529_065416/ppo_continuous_action_v01.deploy_policy.pt",
    map_location="cpu",
    weights_only=False,
)
# agent.eval()

reward_total = 0

for i in range(sim_length_steps):
    acc_sp = agent.infer(observation)
    observation, reward, terminated, truncated, info = env.step(acc_sp)
    observation_history[i] = observation
    reward_total += reward
    
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
