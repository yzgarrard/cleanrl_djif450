import unittest

import gymnasium as gym
import numpy as np
import torch

import custom_envs
from cleanrl_drone.sac_continuous_action_mujoco_v01 import (
    Actor,
    REWARD_TERM_NAMES,
    accumulate_reward_terms,
    get_actor_statistics,
    run_deterministic_eval,
)


class TestSacDroneDiagnostics(unittest.TestCase):
    def setUp(self):
        self.envs = gym.vector.SyncVectorEnv(
            [lambda: gym.make("custom_envs/TacDroneHover-v4")]
        )
        self.actor = Actor(self.envs)

    def tearDown(self):
        self.envs.close()

    def test_actor_paths_have_expected_shapes_and_bounds(self):
        observations = torch.zeros((3, *self.envs.single_observation_space.shape))

        sampled_action, log_prob, sampled_mean = self.actor.get_action(observations)
        deterministic_action_1 = self.actor.get_deterministic_action(observations)
        deterministic_action_2 = self.actor.get_deterministic_action(observations)

        self.assertEqual(sampled_action.shape, (3, 4))
        self.assertEqual(sampled_mean.shape, (3, 4))
        self.assertEqual(log_prob.shape, (3, 1))
        self.assertEqual(deterministic_action_1.shape, (3, 4))
        self.assertTrue(torch.equal(deterministic_action_1, deterministic_action_2))
        self.assertTrue(torch.all(deterministic_action_1 <= 1.0))
        self.assertTrue(torch.all(deterministic_action_1 >= -1.0))

    def test_actor_statistics_are_finite_and_bounded(self):
        observations = torch.zeros((8, *self.envs.single_observation_space.shape))
        rng_state = torch.random.get_rng_state()
        statistics = get_actor_statistics(self.actor, observations)

        self.assertTrue(torch.equal(rng_state, torch.random.get_rng_state()))
        self.assertTrue(all(np.isfinite(value) for value in statistics.values()))
        self.assertGreaterEqual(statistics["sampled_action_saturation_fraction"], 0.0)
        self.assertLessEqual(statistics["sampled_action_saturation_fraction"], 1.0)
        self.assertGreaterEqual(statistics["deterministic_action_saturation_fraction"], 0.0)
        self.assertLessEqual(statistics["deterministic_action_saturation_fraction"], 1.0)

    def test_reward_term_aggregation_handles_vector_and_final_info(self):
        normal_terms = {name: 1.0 for name in REWARD_TERM_NAMES}
        final_terms = {name: 2.0 for name in REWARD_TERM_NAMES}
        infos = {
            "reward_terms": np.asarray([normal_terms], dtype=object),
            "_reward_terms": np.asarray([True]),
            "final_info": np.asarray([{"reward_terms": final_terms}], dtype=object),
        }
        sums = {name: 0.0 for name in REWARD_TERM_NAMES}

        count = accumulate_reward_terms(infos, sums)

        self.assertEqual(count, 2)
        for name in REWARD_TERM_NAMES:
            with self.subTest(reward_term=name):
                self.assertEqual(sums[name], 3.0)

    def test_deterministic_eval_does_not_modify_training_env_or_actor(self):
        training_env = self.envs.envs[0].unwrapped
        training_env.reset(seed=123)
        qpos_before = training_env.data.qpos.copy()
        step_count_before = training_env._step_count
        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in self.actor.state_dict().items()
        }

        returns, lengths, max_episode_steps = run_deterministic_eval(
            actor=self.actor,
            env_id="custom_envs/TacDroneHover-v4",
            device=torch.device("cpu"),
            eval_episodes=1,
            eval_seed=456,
        )

        self.assertEqual(returns.shape, (1,))
        self.assertEqual(lengths.shape, (1,))
        self.assertEqual(max_episode_steps, 1000)
        np.testing.assert_array_equal(training_env.data.qpos, qpos_before)
        self.assertEqual(training_env._step_count, step_count_before)
        for name, parameter in self.actor.state_dict().items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(parameter, actor_before[name]))


if __name__ == "__main__":
    unittest.main()
