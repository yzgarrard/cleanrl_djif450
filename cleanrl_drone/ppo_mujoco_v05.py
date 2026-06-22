# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import copy
import os
import random
import time
from collections import Counter
from dataclasses import dataclass

import custom_envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl_drone.deploy_policy import DronePolicy


ACTION_NAMES = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")
TERMINATION_REASONS = (
    "ground_contact",
    "altitude_error",
    "xy_bounds",
    "excessive_tilt",
    "time_limit",
    "unknown",
)
REWARD_TERM_NAMES = (
    "alive",
    "z",
    "xy",
    "vel",
    "ang",
    "tilt",
    "yaw",
    "act",
    "act_delta",
    "termination",
    "total",
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 8
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of the wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model and deployment artifacts"""
    upload_model: bool = False
    """whether to upload the saved model to Hugging Face"""
    hf_entity: str = ""
    """the user or org name of the model repository"""

    env_id: str = "custom_envs/TacDroneHover-v4"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps per environment per rollout"""
    anneal_lr: bool = True
    """whether to linearly anneal the learning rate"""
    gamma: float = 0.99
    """the discount factor"""
    gae_lambda: float = 0.95
    """the GAE lambda"""
    num_minibatches: int = 64
    """the number of minibatches"""
    update_epochs: int = 10
    """the number of PPO update epochs"""
    norm_adv: bool = True
    """whether to normalize advantages"""
    clip_coef: float = 0.2
    """the PPO clipping coefficient"""
    clip_vloss: bool = True
    """whether to use clipped value loss"""
    ent_coef: float = 0.0
    """coefficient of the sample-based transformed entropy estimate"""
    vf_coef: float = 0.5
    """coefficient of the value loss"""
    max_grad_norm: float = 0.5
    """maximum gradient norm"""
    target_kl: float = None
    """optional target KL threshold"""
    actor_logstd_init: float = 0.0
    """initial pre-tanh Gaussian log standard deviation"""
    actor_logstd_min: float = -3.0
    """minimum pre-tanh Gaussian log standard deviation"""
    actor_logstd_max: float = 3.0
    """maximum pre-tanh Gaussian log standard deviation"""
    deterministic_eval_interval: int = 100_000
    """run deterministic evaluation every N training steps; set <= 0 to disable"""
    deterministic_eval_episodes: int = 5
    """number of fixed-seed deterministic evaluation episodes"""
    deterministic_eval_seed: int = 10_000
    """base seed reused at every deterministic evaluation"""
    final_deterministic_eval_episodes: int = 10
    """number of final deterministic evaluation episodes"""
    final_deterministic_eval_seed: int = 20_000
    """base seed for final deterministic evaluation"""
    deterministic_eval_settling_seconds: float = 5.0
    """initial duration excluded from deterministic tracking metrics"""

    batch_size: int = 0
    """computed rollout batch size"""
    minibatch_size: int = 0
    """computed minibatch size"""
    num_iterations: int = 0
    """computed number of PPO iterations"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_deterministic_eval_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def find_normalize_observation_wrapper(env):
    while True:
        if isinstance(env, gym.wrappers.NormalizeObservation):
            return env
        if not hasattr(env, "env"):
            raise RuntimeError("NormalizeObservation wrapper not found")
        env = env.env


def normalize_eval_obs(obs, obs_mean, obs_var, obs_epsilon, device):
    obs = (obs - obs_mean) / np.sqrt(obs_var + obs_epsilon)
    obs = np.clip(obs, -10.0, 10.0)
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def attitude_error_from_identity(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return float(2.0 * np.arccos(np.clip(np.abs(quat[0]), 0.0, 1.0)))


def run_deterministic_eval(
    agent,
    env_id,
    obs_mean,
    obs_var,
    obs_epsilon,
    device,
    eval_episodes,
    eval_seed,
    settling_seconds=5.0,
):
    eval_env = make_deterministic_eval_env(env_id)
    max_episode_steps = getattr(eval_env.unwrapped, "max_episode_steps", None)
    settling_steps = int(np.ceil(settling_seconds / eval_env.unwrapped.dt))
    episodic_returns = []
    episodic_lengths = []
    termination_counts = Counter()
    metric_sums = {
        "position_error_squared": 0.0,
        "horizontal_error_squared": 0.0,
        "vertical_error_squared": 0.0,
        "velocity_squared": 0.0,
        "attitude_error_squared": 0.0,
        "action_delta_squared": 0.0,
    }
    action_saturation_counts = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    max_position_error = 0.0
    metric_sample_count = 0

    try:
        for episode_idx in range(eval_episodes):
            obs, _ = eval_env.reset(seed=eval_seed + episode_idx)
            terminated = truncated = False
            episodic_return = 0.0
            episodic_length = 0
            previous_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            final_info = {}

            while not (terminated or truncated):
                with torch.no_grad():
                    normalized_obs = normalize_eval_obs(
                        obs, obs_mean, obs_var, obs_epsilon, device
                    )
                    action = agent.get_deterministic_action(normalized_obs)
                action_np = action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, final_info = eval_env.step(action_np)
                episodic_return += float(reward)
                episodic_length += 1

                if episodic_length > settling_steps:
                    unwrapped = eval_env.unwrapped
                    position_error_vector = unwrapped.data.qpos[:3] - unwrapped.pos_des
                    position_error = float(np.linalg.norm(position_error_vector))
                    horizontal_error = float(np.linalg.norm(position_error_vector[:2]))
                    vertical_error = float(abs(position_error_vector[2]))
                    velocity = float(np.linalg.norm(unwrapped.data.qvel[:3]))
                    attitude_error = attitude_error_from_identity(unwrapped.data.qpos[3:7])
                    action_delta = float(np.linalg.norm(action_np - previous_action))

                    metric_sums["position_error_squared"] += position_error**2
                    metric_sums["horizontal_error_squared"] += horizontal_error**2
                    metric_sums["vertical_error_squared"] += vertical_error**2
                    metric_sums["velocity_squared"] += velocity**2
                    metric_sums["attitude_error_squared"] += attitude_error**2
                    metric_sums["action_delta_squared"] += action_delta**2
                    action_saturation_counts += np.abs(action_np) >= 0.95
                    max_position_error = max(max_position_error, position_error)
                    metric_sample_count += 1
                previous_action = action_np.copy()

            episodic_returns.append(episodic_return)
            episodic_lengths.append(episodic_length)
            termination_reason = final_info.get("termination_reason")
            if termination_reason is None:
                termination_reason = "time_limit" if truncated else "unknown"
            termination_counts[termination_reason] += 1
    finally:
        eval_env.close()

    if metric_sample_count:
        eval_metrics = {
            "position_error_rms": np.sqrt(
                metric_sums["position_error_squared"] / metric_sample_count
            ),
            "position_error_max": max_position_error,
            "horizontal_error_rms": np.sqrt(
                metric_sums["horizontal_error_squared"] / metric_sample_count
            ),
            "vertical_error_rms": np.sqrt(
                metric_sums["vertical_error_squared"] / metric_sample_count
            ),
            "velocity_rms": np.sqrt(
                metric_sums["velocity_squared"] / metric_sample_count
            ),
            "attitude_error_rms_deg": np.rad2deg(
                np.sqrt(metric_sums["attitude_error_squared"] / metric_sample_count)
            ),
            "action_delta_rms": np.sqrt(
                metric_sums["action_delta_squared"] / metric_sample_count
            ),
        }
        for idx, action_name in enumerate(ACTION_NAMES):
            eval_metrics[f"{action_name}_saturation_fraction"] = (
                action_saturation_counts[idx] / metric_sample_count
            )
    else:
        eval_metrics = {
            "position_error_rms": np.nan,
            "position_error_max": np.nan,
            "horizontal_error_rms": np.nan,
            "vertical_error_rms": np.nan,
            "velocity_rms": np.nan,
            "attitude_error_rms_deg": np.nan,
            "action_delta_rms": np.nan,
        }
        for action_name in ACTION_NAMES:
            eval_metrics[f"{action_name}_saturation_fraction"] = np.nan
    eval_metrics["tracking_sample_count"] = metric_sample_count
    for reason in TERMINATION_REASONS:
        eval_metrics[f"termination/{reason}_fraction"] = (
            termination_counts[reason] / eval_episodes
        )

    return (
        np.asarray(episodic_returns, dtype=np.float32),
        np.asarray(episodic_lengths, dtype=np.float32),
        max_episode_steps,
        eval_metrics,
    )


def log_deterministic_eval(
    writer,
    prefix,
    eval_returns,
    eval_lengths,
    eval_max_episode_steps,
    eval_metrics,
    global_step,
    log_episode_lengths=False,
):
    if eval_max_episode_steps is None:
        full_length_fraction = np.nan
    else:
        full_length_fraction = np.mean(eval_lengths >= eval_max_episode_steps)
    writer.add_scalar(f"{prefix}/episodic_return_mean", eval_returns.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/episodic_return_min", eval_returns.min().item(), global_step)
    writer.add_scalar(f"{prefix}/episodic_return_max", eval_returns.max().item(), global_step)
    writer.add_scalar(f"{prefix}/episodic_length_mean", eval_lengths.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/full_length_fraction", float(full_length_fraction), global_step)
    for name, value in eval_metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, global_step)
    for idx, (episodic_return, episodic_length) in enumerate(zip(eval_returns, eval_lengths)):
        writer.add_scalar(f"{prefix}/episodic_return", episodic_return.item(), global_step + idx)
        if log_episode_lengths:
            writer.add_scalar(f"{prefix}/episodic_length", episodic_length.item(), global_step + idx)
    return full_length_fraction


def accumulate_reward_terms(infos, reward_term_sums):
    reward_term_count = 0

    def add_terms(reward_terms):
        if not reward_terms:
            return 0
        for name in REWARD_TERM_NAMES:
            reward_term_sums[name] += float(reward_terms.get(name, 0.0))
        return 1

    if "reward_terms" in infos:
        reward_terms_batch = infos["reward_terms"]
        reward_terms_mask = infos.get(
            "_reward_terms", np.ones(len(reward_terms_batch), dtype=bool)
        )
        for reward_terms, has_reward_terms in zip(reward_terms_batch, reward_terms_mask):
            if has_reward_terms:
                reward_term_count += add_terms(reward_terms)
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "reward_terms" in info:
                reward_term_count += add_terms(info["reward_terms"])
    return reward_term_count


def accumulate_episode_end_reasons(infos, episode_end_counts):
    completed_episodes = 0
    if "final_info" not in infos:
        return completed_episodes
    for info in infos["final_info"]:
        if not info:
            continue
        reason = info.get("termination_reason") or "unknown"
        episode_end_counts[reason] = episode_end_counts.get(reason, 0) + 1
        completed_episodes += 1
    return completed_episodes


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        actor_logstd_init=0.0,
        actor_logstd_min=-3.0,
        actor_logstd_max=3.0,
    ):
        super().__init__()
        observation_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.full((1, action_dim), actor_logstd_init)
        )
        self.actor_logstd_min = actor_logstd_min
        self.actor_logstd_max = actor_logstd_max
        action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32)
        if not torch.all(torch.isfinite(action_low)) or not torch.all(
            torch.isfinite(action_high)
        ):
            raise ValueError("Tanh-squashed PPO requires finite action bounds")
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def get_value(self, x):
        return self.critic(x)

    def get_distribution(self, x):
        mean = self.actor_mean(x)
        logstd = torch.clamp(
            self.actor_logstd,
            self.actor_logstd_min,
            self.actor_logstd_max,
        ).expand_as(mean)
        return Normal(mean, logstd.exp()), logstd

    def squash_action(self, latent_action):
        return torch.tanh(latent_action) * self.action_scale + self.action_bias

    def transformed_log_prob(self, distribution, latent_action):
        base_log_prob = distribution.log_prob(latent_action)
        tanh_log_det = 2.0 * (
            np.log(2.0) - latent_action - F.softplus(-2.0 * latent_action)
        )
        scale_log_det = torch.log(self.action_scale)
        return (base_log_prob - tanh_log_det - scale_log_det).sum(1)

    def get_action_and_value(self, x, latent_action=None):
        distribution, _ = self.get_distribution(x)
        if latent_action is None:
            latent_action = distribution.sample()
        action = self.squash_action(latent_action)
        log_prob = self.transformed_log_prob(distribution, latent_action)
        entropy_estimate = -log_prob
        return (
            action,
            latent_action,
            log_prob,
            entropy_estimate,
            self.critic(x),
        )

    def get_deterministic_action(self, x):
        return self.squash_action(self.actor_mean(x))


def get_actor_statistics(agent, observations, latent_actions, actions):
    with torch.no_grad():
        _, logstd = agent.get_distribution(observations)
        action_std = logstd.exp()
        deterministic_actions = agent.get_deterministic_action(observations)
        normalized_actions = (actions - agent.action_bias) / agent.action_scale
        normalized_deterministic = (
            deterministic_actions - agent.action_bias
        ) / agent.action_scale
        statistics = {
            "latent_outside_unit_fraction": (
                latent_actions.abs() > 1.0
            ).float().mean().item(),
            "sampled_action_abs_mean": normalized_actions.abs().mean().item(),
            "sampled_action_saturation_fraction": (
                normalized_actions.abs() >= 0.95
            ).float().mean().item(),
            "deterministic_action_abs_mean": normalized_deterministic.abs().mean().item(),
            "deterministic_action_saturation_fraction": (
                normalized_deterministic.abs() >= 0.95
            ).float().mean().item(),
        }
        for idx, action_name in enumerate(ACTION_NAMES):
            statistics[f"sampled_{action_name}_abs_mean"] = (
                normalized_actions[:, idx].abs().mean().item()
            )
            statistics[f"sampled_{action_name}_saturation_fraction"] = (
                normalized_actions[:, idx].abs() >= 0.95
            ).float().mean().item()
            statistics[f"deterministic_{action_name}_abs_mean"] = (
                normalized_deterministic[:, idx].abs().mean().item()
            )
            statistics[f"deterministic_{action_name}_saturation_fraction"] = (
                normalized_deterministic[:, idx].abs() >= 0.95
            ).float().mean().item()
    return logstd, action_std, statistics


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = f"runs/{run_name}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                idx,
                args.capture_video,
                run_name,
                args.gamma,
            )
            for idx in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    agent = Agent(
        envs,
        args.actor_logstd_init,
        args.actor_logstd_min,
        args.actor_logstd_max,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    latent_actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    actions = torch.zeros_like(latent_actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    next_deterministic_eval_step = args.deterministic_eval_interval
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        reward_term_sums = {name: 0.0 for name in REWARD_TERM_NAMES}
        reward_term_count = 0
        episode_end_counts = Counter()
        completed_episode_count = 0

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                (
                    action,
                    latent_action,
                    logprob,
                    _,
                    value,
                ) = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            latent_actions[step] = latent_action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            reward_term_count += accumulate_reward_terms(infos, reward_term_sums)
            completed_episode_count += accumulate_episode_end_reasons(
                infos, episode_end_counts
            )
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=device
            ).view(-1)
            next_obs = torch.as_tensor(
                next_obs_np, dtype=torch.float32, device=device
            )
            next_done = torch.as_tensor(
                next_done_np, dtype=torch.float32, device=device
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={info['episode']['r']}"
                        )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_latent_actions = latent_actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                (
                    _,
                    _,
                    newlogprob,
                    entropy_estimate,
                    newvalue,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_latent_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - b_returns[mb_inds]
                    ) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss = 0.5 * torch.max(
                        v_loss_unclipped,
                        (v_clipped - b_returns[mb_inds]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * (
                        (newvalue - b_returns[mb_inds]) ** 2
                    ).mean()
                entropy_loss = entropy_estimate.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan
            if var_y == 0
            else 1.0 - np.var(y_true - y_pred) / var_y
        )
        logstd, action_std, actor_statistics = get_actor_statistics(
            agent, b_obs, b_latent_actions, b_actions
        )

        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("charts/actor_logstd_mean", logstd.mean().item(), global_step)
        writer.add_scalar("charts/actor_logstd_min", logstd.min().item(), global_step)
        writer.add_scalar("charts/actor_logstd_max", logstd.max().item(), global_step)
        writer.add_scalar("charts/action_std_mean", action_std.mean().item(), global_step)
        writer.add_scalar("charts/action_std_min", action_std.min().item(), global_step)
        writer.add_scalar("charts/action_std_max", action_std.max().item(), global_step)
        for name in (
            "latent_outside_unit_fraction",
            "sampled_action_abs_mean",
            "sampled_action_saturation_fraction",
            "deterministic_action_abs_mean",
            "deterministic_action_saturation_fraction",
        ):
            writer.add_scalar(f"charts/{name}", actor_statistics[name], global_step)
        for name, value in actor_statistics.items():
            writer.add_scalar(f"actor/{name}", value, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if reward_term_count:
            for name in REWARD_TERM_NAMES:
                writer.add_scalar(
                    f"reward_terms/{name}_mean",
                    reward_term_sums[name] / reward_term_count,
                    global_step,
                )
        writer.add_scalar("episode_end/total", completed_episode_count, global_step)
        for reason in TERMINATION_REASONS:
            count = episode_end_counts[reason]
            fraction = (
                count / completed_episode_count
                if completed_episode_count
                else 0.0
            )
            writer.add_scalar(
                f"episode_end/{reason}_count", count, global_step
            )
            writer.add_scalar(
                f"episode_end/{reason}_fraction",
                fraction,
                global_step,
            )
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

        if (
            args.deterministic_eval_interval > 0
            and args.deterministic_eval_episodes > 0
            and global_step >= next_deterministic_eval_step
        ):
            normalize_wrapper = find_normalize_observation_wrapper(envs.envs[0])
            (
                eval_returns,
                eval_lengths,
                eval_max_episode_steps,
                eval_metrics,
            ) = run_deterministic_eval(
                agent=agent,
                env_id=args.env_id,
                obs_mean=normalize_wrapper.obs_rms.mean.copy(),
                obs_var=normalize_wrapper.obs_rms.var.copy(),
                obs_epsilon=normalize_wrapper.epsilon,
                device=device,
                eval_episodes=args.deterministic_eval_episodes,
                eval_seed=args.deterministic_eval_seed,
                settling_seconds=args.deterministic_eval_settling_seconds,
            )
            full_length_fraction = log_deterministic_eval(
                writer,
                "deterministic_eval",
                eval_returns,
                eval_lengths,
                eval_max_episode_steps,
                eval_metrics,
                global_step,
            )
            print(
                "deterministic_eval:"
                f" global_step={global_step},"
                f" return_mean={eval_returns.mean().item():.3f},"
                f" length_mean={eval_lengths.mean().item():.1f},"
                f" full_length_fraction={float(full_length_fraction):.3f},"
                f" position_error_rms={eval_metrics['position_error_rms']:.4f}"
            )
            while global_step >= next_deterministic_eval_step:
                next_deterministic_eval_step += args.deterministic_eval_interval

    if args.upload_model and args.final_deterministic_eval_episodes <= 0:
        raise ValueError(
            "--upload-model requires --final-deterministic-eval-episodes > 0"
        )

    final_eval_returns = None
    if args.final_deterministic_eval_episodes > 0:
        normalize_wrapper = find_normalize_observation_wrapper(envs.envs[0])
        (
            final_eval_returns,
            final_eval_lengths,
            final_eval_max_episode_steps,
            final_eval_metrics,
        ) = run_deterministic_eval(
            agent=agent,
            env_id=args.env_id,
            obs_mean=normalize_wrapper.obs_rms.mean.copy(),
            obs_var=normalize_wrapper.obs_rms.var.copy(),
            obs_epsilon=normalize_wrapper.epsilon,
            device=device,
            eval_episodes=args.final_deterministic_eval_episodes,
            eval_seed=args.final_deterministic_eval_seed,
            settling_seconds=args.deterministic_eval_settling_seconds,
        )
        final_full_length_fraction = log_deterministic_eval(
            writer,
            "final_deterministic_eval",
            final_eval_returns,
            final_eval_lengths,
            final_eval_max_episode_steps,
            final_eval_metrics,
            global_step,
            log_episode_lengths=True,
        )
        print(
            "final_deterministic_eval:"
            f" global_step={global_step},"
            f" return_mean={final_eval_returns.mean().item():.3f},"
            f" length_mean={final_eval_lengths.mean().item():.1f},"
            f" full_length_fraction={float(final_full_length_fraction):.3f},"
            f" position_error_rms={final_eval_metrics['position_error_rms']:.4f}"
        )

    if args.save_model:
        model_path = f"{run_dir}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        normalize_wrapper = find_normalize_observation_wrapper(envs.envs[0])
        deploy_agent = DronePolicy(
            actor=copy.deepcopy(agent.actor_mean).cpu(),
            obs_mean=normalize_wrapper.obs_rms.mean,
            obs_var=normalize_wrapper.obs_rms.var,
            obs_epsilon=normalize_wrapper.epsilon,
            action_low=envs.single_action_space.low,
            action_high=envs.single_action_space.high,
            squash_actions=True,
        )
        deploy_agent.eval()
        deploy_path = f"{run_dir}/{args.exp_name}.deploy_policy.pt"
        torch.save(deploy_agent, deploy_path)
        print(f"deployment policy saved to {deploy_path}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = (
                f"{args.hf_entity}/{repo_name}"
                if args.hf_entity
                else repo_name
            )
            push_to_hub(
                args,
                final_eval_returns.tolist(),
                repo_id,
                "PPO",
                run_dir,
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
