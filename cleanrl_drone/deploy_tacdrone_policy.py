import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import custom_envs  # noqa: F401 - registers custom Gymnasium environments


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_ID = "custom_envs/TacDroneHover-v4"
DEFAULT_POLICY_GLOB = (
    "runs/custom_envs/TacDroneHover-v4__ppo_mujoco_v04__*/"
    "ppo_mujoco_v04.deploy_policy.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained TacDrone PPO deployment policy in the MuJoCo viewer."
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Path to a ppo_mujoco_v01.deploy_policy.pt artifact. Defaults to the newest TacDrone PPO deploy policy.",
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=3.0,
        help="Seconds to pause after launching the viewer before starting policy rollout.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Throttle simulation to real time. Use --no-realtime to run as fast as possible.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def resolve_policy_path(policy_path: Path | None) -> Path:
    if policy_path is not None:
        resolved_path = policy_path.expanduser()
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Policy file not found: {resolved_path}")
        return resolved_path

    candidates = list(REPO_ROOT.glob(DEFAULT_POLICY_GLOB))
    if not candidates:
        raise FileNotFoundError(
            "No TacDrone deploy policy found. Pass --policy-path or train with "
            "`python cleanrl_drone/ppo_mujoco_v01.py --save-model` first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_policy(policy_path: Path, device: torch.device):
    policy = torch.load(policy_path, map_location=device, weights_only=False)
    policy.to(device)
    policy.eval()
    return policy


def viewer_is_running(env) -> bool:
    viewer = getattr(env.unwrapped, "_viewer", None)
    if viewer is None:
        return True
    if not hasattr(viewer, "is_running"):
        return True
    return viewer.is_running()


def run_episode(
    env,
    policy,
    seed: int,
    episode_idx: int,
    max_steps: int,
    realtime: bool,
    startup_delay: float,
):
    obs, _ = env.reset(seed=seed + episode_idx)
    env.render()
    if startup_delay > 0:
        print(f"Waiting {startup_delay:.1f}s before starting rollout...")
        time.sleep(startup_delay)

    sim_start_time = env.unwrapped.data.time
    wall_start_time = time.time()
    total_reward = 0.0

    for step in range(max_steps):
        if not viewer_is_running(env):
            return total_reward, step, True

        action = np.asarray(policy.infer(obs), dtype=np.float32)
        if action.shape != env.action_space.shape:
            raise ValueError(
                f"Policy returned action shape {action.shape}; expected {env.action_space.shape}"
            )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        env.render()

        if not viewer_is_running(env):
            return total_reward, step + 1, True

        if realtime:
            sim_elapsed = env.unwrapped.data.time - sim_start_time
            wall_elapsed = time.time() - wall_start_time
            sleep_time = sim_elapsed - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if terminated or truncated:
            print(
                f"episode={episode_idx + 1} step={step + 1} "
                f"return={total_reward:.3f} terminated={terminated} truncated={truncated} "
                f"z={info.get('z', float('nan')):.3f} "
                f"tilt_deg={info.get('tilt_deg', float('nan')):.2f}"
            )
            return total_reward, step + 1, False

    print(f"episode={episode_idx + 1} step={max_steps} return={total_reward:.3f} max_steps_reached=True")
    return total_reward, max_steps, False


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but torch.cuda.is_available() is false")

    policy_path = resolve_policy_path(args.policy_path)
    policy = load_policy(policy_path, device)
    print(f"Loaded policy: {policy_path}")
    print("Close the MuJoCo viewer window to stop.")

    env = gym.make(args.env_id, render_mode="human")
    try:
        for episode_idx in range(args.episodes):
            _, _, viewer_closed = run_episode(
                env=env,
                policy=policy,
                seed=args.seed,
                episode_idx=episode_idx,
                max_steps=args.max_steps,
                realtime=args.realtime,
                startup_delay=args.startup_delay,
            )
            if viewer_closed:
                print("Viewer closed; exiting.")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
