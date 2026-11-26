"""Numerical checks for target updates, Double DQN and dueling heads."""
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
import torch
from torch import nn

from dqn_agent import DQNAgent
from dqn_train import TrainConfig, evaluate, train
from q_network import QNetwork, BreakoutQNetwork
from sample import FrameNumpy, SampleBatchNumpy


class TinyQNetwork(QNetwork):
    def __init__(self):
        nn.Module.__init__(self)
        self.seq = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.seq.weight.copy_(torch.tensor([[0., 2.], [0., 1.]]))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.1)


class DQNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def make_agent(self, **kwargs):
        agent = DQNAgent(dict(action_dim=2, gamma=.5, target_update_freq=2, **kwargs))
        agent.set_model(TinyQNetwork())
        with torch.no_grad():
            # Online selects action 0, whereas the target's maximum is action 1.
            agent.target_model.seq.weight.copy_(torch.tensor([[0., 3.], [0., 9.]]))
        return agent

    def batch(self, agent):
        batch = SampleBatchNumpy.stack([
            FrameNumpy.from_dict(dict(obs=np.array([1., 0.], dtype=np.float32),
                next_obs=np.array([0., 1.], dtype=np.float32), action=0,
                reward=1., done=done)) for done in (False, True)])
        agent.sample_process(batch)
        return batch

    def test_double_dqn_targets_terminal_mask_and_sync(self):
        agent = self.make_agent(double_dqn=True)
        original = agent.target_model.seq.weight.detach().clone()
        metrics = agent.learn(self.batch(agent))
        # Nonterminal target: 1 + .5 * 3 = 2.5; terminal target: 1.
        self.assertAlmostEqual(metrics['target_mean'], 1.75)
        self.assertAlmostEqual(metrics['loss'], (2.5 ** 2 + 1) / 2)
        torch.testing.assert_close(agent.target_model.seq.weight, original)
        self.assertFalse(agent.target_model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in agent.target_model.parameters()))
        # Next-state column must have no gradient through the TD target.
        torch.testing.assert_close(agent.model.seq.weight[:, 1], torch.tensor([2., 1.]))
        agent.learn(self.batch(agent))
        torch.testing.assert_close(agent.target_model.seq.weight, agent.model.seq.weight)
        self.assertFalse(torch.equal(agent.target_model.seq.weight, original))

    def test_target_dqn_uses_maximum(self):
        agent = self.make_agent()
        metrics = agent.learn(self.batch(agent))
        self.assertAlmostEqual(metrics['target_mean'], (1 + .5 * 9 + 1) / 2)

    def test_huber_and_gradient_clipping(self):
        agent = self.make_agent(double_dqn=True, huber_loss=True, max_grad_norm=.1)
        metrics = agent.learn(self.batch(agent))
        self.assertAlmostEqual(metrics['loss'], (2. + .5) / 2)
        self.assertAlmostEqual(metrics['grad_norm'], 1.)
        norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in agent.model.parameters()]))
        self.assertLessEqual(norm.item(), .100001)
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.model.parameters()))

    def test_dueling_aggregation_shapes_and_both_heads_learn(self):
        model = BreakoutQNetwork(4, lr=5e-5, hidden_dim=32, dueling=True)
        model.eval()
        self.assertFalse(model.training)
        model.train()
        obs = torch.randint(256, (2, 4, 80, 80), dtype=torch.uint8)
        q = model.inference(obs)
        self.assertEqual(q.shape, (2, 4))
        torch.testing.assert_close(model.inference(obs[0]), q[0])
        features = model.seq(obs.float() / 255.)
        torch.testing.assert_close(q.mean(dim=1, keepdim=True), model.value(features))
        before = [head[-1].weight.detach().clone() for head in (model.value, model.advantage)]
        model.optimize((q[:, 0] - 2).square().mean(), 10.)
        for previous, head in zip(before, (model.value, model.advantage)):
            self.assertFalse(torch.equal(previous, head[-1].weight))

    def test_evaluation_randomness_is_reproducible_and_isolated(self):
        class FakeEnv:
            action_dim = 2
            def reset(self, conf):
                self.steps = 0
                return np.zeros(2, dtype=np.float32)
            def step(self, action):
                self.steps += 1
                return np.zeros(2, dtype=np.float32), float(action), self.steps == 30, {}
        agent = self.make_agent()
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        scores = evaluate(agent, FakeEnv(), range(10), epsilon=.5)
        np.testing.assert_array_equal(np.random.random(5), expected)
        self.assertEqual(scores, evaluate(agent, FakeEnv(), range(10), epsilon=.5))
        self.assertGreater(len(set(scores)), 1)

    def test_training_bootstraps_on_truncation_but_not_termination(self):
        class FakeEnv:
            action_dim = 2
            state_dim = (2,)
            def __init__(self, terminal):
                self.env = self
                self.terminal = terminal
            def reset(self, conf=None):
                self.steps = 0
                return np.zeros(2, dtype=np.float32)
            def step(self, action):
                self.steps += 1
                done = self.steps == 3
                return np.ones(2, dtype=np.float32), 1., done, dict(
                    terminated=done and self.terminal, truncated=done and not self.terminal)
            def close(self):
                pass
        for terminal in (False, True):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as output:
                masks = []
                def learn(agent, batch):
                    masks.extend(batch.done.flatten().tolist())
                    return dict(loss=0., q_mean=0., target_mean=0., grad_norm=None)
                config = TrainConfig(episodes=1, batch_size=2, buffer_size=4,
                                     eval_episodes=1, final_eval_episodes=1)
                # Include all transitions in the test update to check the final mask.
                with patch('dqn_train.make_env', side_effect=lambda c: FakeEnv(terminal)), \
                     patch('dqn_train.random.sample', side_effect=lambda population, k: list(population)), \
                     patch.object(DQNAgent, 'learn', learn):
                    result = train(config, output)
                self.assertEqual(result['env_steps'], 3)
                self.assertEqual(masks, [0., 0., float(terminal)])


if __name__ == '__main__':
    unittest.main()
