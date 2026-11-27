"""Hand-calculated returns and episode boundary checks for n-step replay."""
import unittest

import numpy as np

from n_step import NStepAccumulator
from sample import FrameNumpy


def transition(index, reward, terminal=False):
    return FrameNumpy.from_dict(dict(
        obs=np.array([index], dtype=np.uint8), action=index % 4,
        reward=reward, next_obs=np.array([index + 1], dtype=np.uint8), done=terminal))


class NStepTests(unittest.TestCase):
    def test_three_step_sliding_returns_and_terminal_tails(self):
        accumulator = NStepAccumulator(3, .5)
        self.assertEqual(accumulator.append(transition(0, 1), False), [])
        self.assertEqual(accumulator.append(transition(1, 2), False), [])
        first = accumulator.append(transition(2, 3), False)
        self.assertEqual(len(first), 1)
        samples = first + accumulator.append(transition(3, 4, True), True)
        self.assertEqual([s.reward for s in samples], [2.75, 4.5, 5., 4.])
        self.assertEqual([s.discount for s in samples], [.125, .125, .25, .5])
        self.assertEqual([s.done for s in samples], [False, True, True, True])
        self.assertEqual([s.action for s in samples], [0, 1, 2, 3])
        self.assertEqual([int(s.obs[0]) for s in samples], [0, 1, 2, 3])
        self.assertEqual([int(s.next_obs[0]) for s in samples], [3, 4, 4, 4])
        self.assertTrue(all(s.obs.dtype == np.uint8 for s in samples))
        self.assertFalse(accumulator.pending)

    def test_short_truncated_episode_bootstraps_and_cannot_leak(self):
        accumulator = NStepAccumulator(3, .5)
        accumulator.append(transition(0, 1), False)
        samples = accumulator.append(transition(1, 2), True)
        self.assertEqual([s.reward for s in samples], [2., 2.])
        self.assertEqual([s.discount for s in samples], [.25, .5])
        self.assertEqual([s.done for s in samples], [False, False])
        self.assertEqual([int(s.next_obs[0]) for s in samples], [2, 2])
        self.assertFalse(accumulator.pending)
        next_episode = accumulator.append(transition(10, 100, True), True)
        self.assertEqual(len(next_episode), 1)
        self.assertEqual(next_episode[0].reward, 100.)
        self.assertEqual(next_episode[0].discount, .5)
        self.assertEqual(int(next_episode[0].obs[0]), 10)

    def test_one_step_matches_raw_transitions(self):
        accumulator = NStepAccumulator(1, .99)
        for index in range(4):
            item = transition(index, index + .25, index == 3)
            samples = accumulator.append(item, index == 3)
            self.assertEqual(len(samples), 1)
            for key in ('reward', 'done', 'action'):
                self.assertEqual(samples[0][key], item[key])
            np.testing.assert_array_equal(samples[0].obs, item.obs)
            np.testing.assert_array_equal(samples[0].next_obs, item.next_obs)
            self.assertEqual(samples[0].discount, .99)
            self.assertFalse(accumulator.pending)

    def test_zero_gamma_and_invalid_horizon(self):
        accumulator = NStepAccumulator(3, 0.)
        accumulator.append(transition(0, 1), False)
        samples = accumulator.append(transition(1, 2), True)
        self.assertEqual([s.reward for s in samples], [1., 2.])
        self.assertEqual([s.discount for s in samples], [0., 0.])
        for horizon in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                NStepAccumulator(horizon, .99)


if __name__ == '__main__':
    unittest.main()
