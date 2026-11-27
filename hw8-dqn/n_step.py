"""Build replay transitions with discounted rewards over up to n decisions."""
from collections import deque

from sample import FrameNumpy


class NStepAccumulator:
    def __init__(self, n_step, gamma):
        if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step < 1:
            raise ValueError('n_step must be a positive integer')
        self.n_step = n_step
        self.gamma = gamma
        self.pending = deque()

    def append(self, transition, episode_end):
        """done means true termination; episode_end also includes truncation.

        Emit each starting state exactly once, flushing shorter tails at the
        episode boundary. Truncated tails bootstrap from the last observation.
        """
        self.pending.append(transition)
        ended = episode_end or bool(transition.done)
        ready = []
        while self.pending and (ended or len(self.pending) >= self.n_step):
            reward, discount = 0.0, 1.0
            for item in self.pending:
                reward += discount * float(item.reward)
                discount *= self.gamma
            first, last = self.pending[0], self.pending[-1]
            ready.append(FrameNumpy.from_dict(dict(
                obs=first.obs, action=first.action, reward=reward,
                next_obs=last.next_obs, done=last.done, discount=discount)))
            self.pending.popleft()
        return ready
