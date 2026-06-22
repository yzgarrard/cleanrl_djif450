import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import custom_envs
from cleanrl_drone.deploy_policy import DronePolicy
from cleanrl_drone.ppo_mujoco_v05 import (
    ACTION_NAMES,
    Agent,
    REWARD_TERM_NAMES,
    TERMINATION_REASONS,
    accumulate_episode_end_reasons,
    accumulate_reward_terms,
    run_deterministic_eval,
)


class TestTanhSquashedPPO(unittest.TestCase):
    def setUp(self):
        self.envs = gym.vector.SyncVectorEnv(
            [lambda: gym.make("custom_envs/TacDroneHover-v4")]
        )
        self.agent = Agent(self.envs)
        self.observations = torch.zeros(
            (8, *self.envs.single_observation_space.shape)
        )

    def tearDown(self):
        self.envs.close()

    def test_stochastic_and_deterministic_actions_are_bounded(self):
        action, _, _, _, _ = self.agent.get_action_and_value(self.observations)
        deterministic_1 = self.agent.get_deterministic_action(self.observations)
        deterministic_2 = self.agent.get_deterministic_action(self.observations)

        self.assertTrue(torch.all(action <= 1.0))
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.equal(deterministic_1, deterministic_2))
        self.assertTrue(torch.all(deterministic_1 <= 1.0))
        self.assertTrue(torch.all(deterministic_1 >= -1.0))

    def test_transformed_log_prob_is_finite_near_saturation(self):
        distribution, _ = self.agent.get_distribution(self.observations)
        latent = torch.full((8, 4), 10.0)
        log_prob = self.agent.transformed_log_prob(distribution, latent)

        self.assertTrue(torch.all(torch.isfinite(log_prob)))

    def test_stored_latent_action_recomputes_identical_log_prob(self):
        _, latent, old_log_prob, _, _ = self.agent.get_action_and_value(
            self.observations
        )
        _, _, new_log_prob, _, _ = self.agent.get_action_and_value(
            self.observations, latent
        )

        torch.testing.assert_close(old_log_prob, new_log_prob)
        torch.testing.assert_close(
            (new_log_prob - old_log_prob).exp(),
            torch.ones_like(old_log_prob),
        )

    def test_entropy_estimate_and_gradients_are_finite(self):
        _, _, log_prob, entropy, value = self.agent.get_action_and_value(
            self.observations
        )
        loss = -(log_prob.mean() + 0.01 * entropy.mean()) + value.mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for parameter in self.agent.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.all(torch.isfinite(parameter.grad)))

    def test_reward_and_episode_end_aggregation(self):
        normal_terms = {name: 1.0 for name in REWARD_TERM_NAMES}
        final_terms = {name: 2.0 for name in REWARD_TERM_NAMES}
        infos = {
            "reward_terms": np.asarray([normal_terms], dtype=object),
            "_reward_terms": np.asarray([True]),
            "final_info": np.asarray(
                [
                    {
                        "reward_terms": final_terms,
                        "termination_reason": "xy_bounds",
                    }
                ],
                dtype=object,
            ),
        }
        sums = {name: 0.0 for name in REWARD_TERM_NAMES}
        counts = {}

        reward_count = accumulate_reward_terms(infos, sums)
        episode_count = accumulate_episode_end_reasons(infos, counts)

        self.assertEqual(reward_count, 2)
        self.assertEqual(episode_count, 1)
        self.assertEqual(counts["xy_bounds"], 1)
        for name in REWARD_TERM_NAMES:
            self.assertEqual(sums[name], 3.0)

    def test_deterministic_evaluation_metrics(self):
        returns, lengths, max_steps, metrics = run_deterministic_eval(
            agent=self.agent,
            env_id="custom_envs/TacDroneHover-v4",
            obs_mean=np.zeros(20, dtype=np.float32),
            obs_var=np.ones(20, dtype=np.float32),
            obs_epsilon=1e-8,
            device=torch.device("cpu"),
            eval_episodes=1,
            eval_seed=123,
            settling_seconds=0.0,
        )

        self.assertEqual(returns.shape, (1,))
        self.assertEqual(lengths.shape, (1,))
        self.assertEqual(max_steps, 1000)
        self.assertIn("position_error_rms", metrics)
        for action_name in ACTION_NAMES:
            self.assertIn(f"{action_name}_saturation_fraction", metrics)
        self.assertAlmostEqual(
            sum(
                metrics[f"termination/{reason}_fraction"]
                for reason in TERMINATION_REASONS
            ),
            1.0,
        )


class TestDronePolicySquashing(unittest.TestCase):
    def make_policy(self, squash_actions):
        actor = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            actor.weight.fill_(2.0)
        return DronePolicy(
            actor=actor,
            obs_mean=np.zeros(2, dtype=np.float32),
            obs_var=np.ones(2, dtype=np.float32),
            obs_epsilon=1e-8,
            action_low=np.array([-1.0], dtype=np.float32),
            action_high=np.array([1.0], dtype=np.float32),
            squash_actions=squash_actions,
        )

    def test_old_policy_behavior_remains_clamped(self):
        policy = self.make_policy(False)
        action = policy.infer(np.ones(2, dtype=np.float32))
        np.testing.assert_allclose(action, [1.0])

    def test_new_policy_behavior_is_tanh_squashed(self):
        policy = self.make_policy(True)
        action = policy.infer(np.ones(2, dtype=np.float32))
        np.testing.assert_allclose(action, [np.tanh(4.0)], rtol=1e-6)

    def test_missing_squash_attribute_defaults_to_old_behavior(self):
        policy = self.make_policy(False)
        del policy.squash_actions
        action = policy.infer(np.ones(2, dtype=np.float32))
        np.testing.assert_allclose(action, [1.0])


if __name__ == "__main__":
    unittest.main()
