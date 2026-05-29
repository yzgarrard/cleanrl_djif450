import numpy as np
import torch
import torch.nn as nn


class DronePolicy(nn.Module):
    def __init__(self, actor, obs_mean, obs_var, obs_epsilon, action_low, action_high):
        super().__init__()
        self.actor = actor
        self.obs_epsilon = float(obs_epsilon)
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_var", torch.as_tensor(obs_var, dtype=torch.float32))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    @torch.no_grad()
    def infer(self, obs):
        single_obs = np.asarray(obs).ndim == 1
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.obs_mean.device)
        if single_obs:
            obs = obs.unsqueeze(0)

        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.obs_epsilon)
        obs = torch.clamp(obs, -10.0, 10.0)
        action = self.actor(obs)
        action = torch.clamp(action, self.action_low, self.action_high)

        if single_obs:
            action = action.squeeze(0)
        return action.cpu().numpy()
