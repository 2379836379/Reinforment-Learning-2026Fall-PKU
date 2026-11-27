"""Integration checks against the installed ALE Breakout environment."""
import unittest

import numpy as np
import torch

from dqn_agent import DQNAgent
from dqn_train import TrainConfig, evaluate, make_env
from gym_env import BreakoutEnv
from q_network import BreakoutQNetwork, QNetwork
from sample import FrameNumpy, SampleBatchNumpy


class BreakoutIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = BreakoutEnv(max_episode_steps=12)
        self.addCleanup(self.env.env.close)

    def test_stacked_observations_seed_fire_and_time_limit(self):
        obs = self.env.reset({'seed': 11})
        original = obs.copy()
        self.assertEqual(obs.shape, (4, 80, 80))
        self.assertEqual(obs.dtype, np.uint8)
        np.testing.assert_array_equal(obs[0], obs[-1])
        next_obs, _, done, info = self.env.step(0)
        self.assertEqual(info['executed_action'], self.env.fire_action)
        np.testing.assert_array_equal(next_obs[:3], original[1:])
        np.testing.assert_array_equal(obs, original)
        self.assertFalse(done)
        for _ in range(11):
            _, _, done, info = self.env.step(0)
        self.assertTrue(done)
        self.assertTrue(info['truncated'])
        self.assertFalse(info['terminated'])
        np.testing.assert_array_equal(self.env.reset({'seed': 11}), original)

    def test_cnn_single_batch_and_dqn_update(self):
        torch.manual_seed(11)
        model = BreakoutQNetwork(self.env.action_dim, lr=1e-4, hidden_dim=64)
        agent = DQNAgent(dict(action_dim=self.env.action_dim))
        agent.set_model(model)
        obs = self.env.reset({'seed': 11})
        samples = []
        for _ in range(8):
            action = agent.predict(obs)
            nxt, reward, done, info = self.env.step(action)
            samples.append(FrameNumpy.from_dict(dict(obs=obs, next_obs=nxt,
                action=info['executed_action'], reward=reward, done=done)))
            obs = nxt
        self.assertEqual(model.inference(torch.from_numpy(obs)).shape, (4,))
        batch = SampleBatchNumpy.stack(samples)
        self.assertEqual(model.inference(torch.from_numpy(batch.obs)).shape, (8, 4))
        before = model.seq[-1].weight.detach().clone()
        agent.sample_process(batch)
        agent.learn(batch)
        self.assertFalse(torch.equal(before, model.seq[-1].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        # Evaluation must reset through the wrapper, keeping preprocessing intact.
        self.assertEqual(len(evaluate(agent, self.env, [10000, 10001])), 2)

    def test_fire_after_losing_a_life(self):
        env = BreakoutEnv(max_episode_steps=1000)
        self.addCleanup(env.env.close)
        env.reset({'seed': 11})
        initial_lives = env.lives
        for _ in range(999):
            _, _, done, info = env.step(0)
            if info['lives'] < initial_lives:
                self.assertFalse(done)
                _, _, _, info = env.step(0)
                self.assertEqual(info['executed_action'], env.fire_action)
                break
        else:
            self.fail('Expected the stationary paddle to lose a life')

    def test_cartpole_path_remains_vector_based(self):
        env = make_env(TrainConfig())
        self.addCleanup(env.env.close)
        obs = env.reset({'seed': 11})
        self.assertEqual(obs.shape, (4,))
        model = QNetwork(4, 2, lr=5e-4)
        agent = DQNAgent(dict(action_dim=2))
        agent.set_model(model)
        self.assertEqual(model.inference(torch.from_numpy(obs)).shape, (2,))
        self.assertEqual(len(evaluate(agent, env, [10000])), 1)


if __name__ == '__main__':
    unittest.main()
