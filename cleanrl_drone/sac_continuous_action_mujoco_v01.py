# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from collections import Counter
from dataclasses import dataclass

import gymnasium as gym
import custom_envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


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
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "custom_envs/TacDroneHover-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    deterministic_eval_interval: int = 50_000
    """run deployment-style deterministic evaluation every N training steps; set <= 0 to disable"""
    deterministic_eval_episodes: int = 20
    """the number of deterministic evaluation episodes to run each time"""
    deterministic_eval_seed: int = 10_000
    """the base seed for deterministic evaluation episodes"""
    final_deterministic_eval_episodes: int = 20
    """the number of deployment-style deterministic evaluation episodes to run after training"""
    final_deterministic_eval_seed: int = 20_000
    """the base seed for final deterministic evaluation episodes"""
    deterministic_eval_settling_seconds: float = 5.0
    """exclude this initial duration from deterministic tracking metrics"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def make_deterministic_eval_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            64,
        )
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(64, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_deterministic_action(self, x):
        mean, _ = self(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias


ACTION_NAMES = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")
TERMINATION_REASONS = (
    "ground_contact",
    "altitude_error",
    "xy_bounds",
    "excessive_tilt",
    "time_limit",
    "unknown",
)


def attitude_error_from_identity(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return float(2.0 * np.arccos(np.clip(np.abs(quat[0]), 0.0, 1.0)))


def run_deterministic_eval(
    actor,
    env_id,
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
            terminated = False
            truncated = False
            episodic_return = 0.0
            episodic_length = 0
            previous_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            final_info = {}

            while not (terminated or truncated):
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor.get_deterministic_action(obs_tensor)

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
        reward_terms_mask = infos.get("_reward_terms", np.ones(len(reward_terms_batch), dtype=bool))
        for reward_terms, has_reward_terms in zip(reward_terms_batch, reward_terms_mask):
            if has_reward_terms:
                reward_term_count += add_terms(reward_terms)

    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "reward_terms" in info:
                reward_term_count += add_terms(info["reward_terms"])

    return reward_term_count


def get_actor_statistics(actor, observations):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if observations.is_cuda else None
    try:
        with torch.no_grad():
            _, log_std = actor(observations)
            action_std = log_std.exp()
            sampled_action, sampled_log_prob, deterministic_action = actor.get_action(observations)

            sampled_action_normalized = (sampled_action - actor.action_bias) / actor.action_scale
            deterministic_action_normalized = (
                deterministic_action - actor.action_bias
            ) / actor.action_scale
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    statistics = {
        "actor_logstd_mean": log_std.mean().item(),
        "actor_logstd_min": log_std.min().item(),
        "actor_logstd_max": log_std.max().item(),
        "action_std_mean": action_std.mean().item(),
        "action_std_min": action_std.min().item(),
        "action_std_max": action_std.max().item(),
        "policy_entropy": -sampled_log_prob.mean().item(),
        "sampled_action_abs_mean": sampled_action_normalized.abs().mean().item(),
        "sampled_action_saturation_fraction": (
            sampled_action_normalized.abs() >= 0.95
        ).float().mean().item(),
        "deterministic_action_abs_mean": deterministic_action_normalized.abs().mean().item(),
        "deterministic_action_saturation_fraction": (
            deterministic_action_normalized.abs() >= 0.95
        ).float().mean().item(),
    }
    for idx, action_name in enumerate(ACTION_NAMES):
        statistics[f"sampled_{action_name}_abs_mean"] = (
            sampled_action_normalized[:, idx].abs().mean().item()
        )
        statistics[f"sampled_{action_name}_saturation_fraction"] = (
            sampled_action_normalized[:, idx].abs() >= 0.95
        ).float().mean().item()
        statistics[f"deterministic_{action_name}_abs_mean"] = (
            deterministic_action_normalized[:, idx].abs().mean().item()
        )
        statistics[f"deterministic_{action_name}_saturation_fraction"] = (
            deterministic_action_normalized[:, idx].abs() >= 0.95
        ).float().mean().item()
    return statistics


def maybe_save_best_actor(
    actor,
    run_dir,
    global_step,
    full_length_fraction,
    eval_metrics,
    best_position_error_rms,
):
    position_error_rms = eval_metrics["position_error_rms"]
    if (
        full_length_fraction < 1.0
        or not np.isfinite(position_error_rms)
        or position_error_rms >= best_position_error_rms
    ):
        return best_position_error_rms, None

    checkpoint_path = os.path.join(run_dir, "best_actor.cleanrl_model")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "global_step": global_step,
            "full_length_fraction": float(full_length_fraction),
            "position_error_rms": float(position_error_rms),
            "eval_metrics": eval_metrics,
        },
        checkpoint_path,
    )
    return float(position_error_rms), checkpoint_path


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    next_deterministic_eval_step = args.deterministic_eval_interval
    reward_term_sums = {name: 0.0 for name in REWARD_TERM_NAMES}
    reward_term_count = 0
    best_position_error_rms = np.inf

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        reward_term_count += accumulate_reward_terms(infos, reward_term_sums)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                actor_statistics = get_actor_statistics(actor, data.observations)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                for name, value in actor_statistics.items():
                    writer.add_scalar(f"actor/{name}", value, global_step)
                if reward_term_count > 0:
                    for name in REWARD_TERM_NAMES:
                        writer.add_scalar(
                            f"reward_terms/{name}_mean",
                            reward_term_sums[name] / reward_term_count,
                            global_step,
                        )
                    reward_term_sums = {name: 0.0 for name in REWARD_TERM_NAMES}
                    reward_term_count = 0
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        completed_steps = global_step + 1
        if (
            args.deterministic_eval_interval > 0
            and args.deterministic_eval_episodes > 0
            and completed_steps >= next_deterministic_eval_step
        ):
            (
                eval_returns,
                eval_lengths,
                eval_max_episode_steps,
                eval_metrics,
            ) = run_deterministic_eval(
                actor=actor,
                env_id=args.env_id,
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
                completed_steps,
            )
            best_position_error_rms, checkpoint_path = maybe_save_best_actor(
                actor=actor,
                run_dir=run_dir,
                global_step=completed_steps,
                full_length_fraction=full_length_fraction,
                eval_metrics=eval_metrics,
                best_position_error_rms=best_position_error_rms,
            )
            if checkpoint_path is not None:
                writer.add_scalar(
                    "deterministic_eval/best_position_error_rms",
                    best_position_error_rms,
                    completed_steps,
                )
                print(f"best actor saved to {checkpoint_path}")
            print(
                "deterministic_eval:"
                f" global_step={completed_steps},"
                f" return_mean={eval_returns.mean().item():.3f},"
                f" length_mean={eval_lengths.mean().item():.1f},"
                f" full_length_fraction={float(full_length_fraction):.3f},"
                f" position_error_rms={eval_metrics['position_error_rms']:.4f}"
            )
            while completed_steps >= next_deterministic_eval_step:
                next_deterministic_eval_step += args.deterministic_eval_interval

    if args.final_deterministic_eval_episodes > 0:
        (
            final_eval_returns,
            final_eval_lengths,
            final_eval_max_episode_steps,
            final_eval_metrics,
        ) = run_deterministic_eval(
            actor=actor,
            env_id=args.env_id,
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
            args.total_timesteps,
            log_episode_lengths=True,
        )
        print(
            "final_deterministic_eval:"
            f" global_step={args.total_timesteps},"
            f" return_mean={final_eval_returns.mean().item():.3f},"
            f" length_mean={final_eval_lengths.mean().item():.1f},"
            f" full_length_fraction={float(final_full_length_fraction):.3f},"
            f" position_error_rms={final_eval_metrics['position_error_rms']:.4f}"
        )

    envs.close()
    writer.close()
