import unittest

import gymnasium as gym
import numpy as np
import torch

import custom_envs
from cleanrl_drone.sac_continuous_action_mujoco_v01 import (
    Actor,
    ACTION_NAMES,
    REWARD_TERM_NAMES,
    accumulate_reward_terms,
    get_actor_statistics,
    maybe_save_best_actor,
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
        for action_name in ACTION_NAMES:
            self.assertIn(
                f"sampled_{action_name}_saturation_fraction",
                statistics,
            )
            self.assertIn(
                f"deterministic_{action_name}_saturation_fraction",
                statistics,
            )

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

        returns, lengths, max_episode_steps, metrics = run_deterministic_eval(
            actor=self.actor,
            env_id="custom_envs/TacDroneHover-v4",
            device=torch.device("cpu"),
            eval_episodes=1,
            eval_seed=456,
            settling_seconds=0.0,
        )

        self.assertEqual(returns.shape, (1,))
        self.assertEqual(lengths.shape, (1,))
        self.assertEqual(max_episode_steps, 1000)
        self.assertIn("position_error_rms", metrics)
        self.assertIn("termination/time_limit_fraction", metrics)
        for action_name in ACTION_NAMES:
            self.assertIn(f"{action_name}_saturation_fraction", metrics)
        termination_fraction = sum(
            value
            for name, value in metrics.items()
            if name.startswith("termination/")
        )
        self.assertAlmostEqual(termination_fraction, 1.0)
        np.testing.assert_array_equal(training_env.data.qpos, qpos_before)
        self.assertEqual(training_env._step_count, step_count_before)
        for name, parameter in self.actor.state_dict().items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(parameter, actor_before[name]))

    def test_best_actor_requires_full_length_and_improves_tracking(self):
        import tempfile

        with tempfile.TemporaryDirectory() as run_dir:
            best, path = maybe_save_best_actor(
                actor=self.actor,
                run_dir=run_dir,
                global_step=100,
                full_length_fraction=0.95,
                eval_metrics={"position_error_rms": 0.1},
                best_position_error_rms=np.inf,
            )
            self.assertTrue(np.isinf(best))
            self.assertIsNone(path)

            best, path = maybe_save_best_actor(
                actor=self.actor,
                run_dir=run_dir,
                global_step=200,
                full_length_fraction=1.0,
                eval_metrics={"position_error_rms": 0.1},
                best_position_error_rms=best,
            )
            self.assertEqual(best, 0.1)
            self.assertIsNotNone(path)

            unchanged_best, path = maybe_save_best_actor(
                actor=self.actor,
                run_dir=run_dir,
                global_step=300,
                full_length_fraction=1.0,
                eval_metrics={"position_error_rms": 0.2},
                best_position_error_rms=best,
            )
            self.assertEqual(unchanged_best, best)
            self.assertIsNone(path)


class TestTacDroneControllerAndReward(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("custom_envs/TacDroneHover-v4").unwrapped
        self.env.reset(seed=123)

    def tearDown(self):
        self.env.close()

    def test_hover_centered_action_penalty(self):
        self.env.last_action = self.env.hover_action.copy()
        _, hover_terms = self.env._compute_reward(self.env.hover_action.copy())
        _, saturated_terms = self.env._compute_reward(np.ones(4, dtype=np.float32))

        self.assertAlmostEqual(hover_terms["act"], 0.0)
        self.assertLess(saturated_terms["act"], 0.0)

    def test_rate_integrators_are_clipped(self):
        self.env.rollrate_err_accum = 1e6
        self.env.pitchrate_err_accum = -1e6
        self.env.yawrate_err_accum = 1e6
        self.env.step(self.env.hover_action.copy())
        limits = self.env.max_i_torque / np.array(
            [
                self.env.MC_ROLLRATE_I,
                self.env.MC_PITCHRATE_I,
                self.env.MC_YAWRATE_I,
            ]
        )

        self.assertLessEqual(abs(self.env.rollrate_err_accum), limits[0])
        self.assertLessEqual(abs(self.env.pitchrate_err_accum), limits[1])
        self.assertLessEqual(abs(self.env.yawrate_err_accum), limits[2])


if __name__ == "__main__":
    unittest.main()
