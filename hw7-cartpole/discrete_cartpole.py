"""n-step SARSA on a discretized CartPole environment.

Run ``python3 discrete_cartpole.py --n-values 1 3 5 --seeds 0 1 2`` to
compare three n values and write per-seed results to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore[no-redef]

import numpy as np

State = tuple[int, int, int, int]
Transition = tuple[State, int, float, State, bool, bool, int | None]


class DiscreteCartPoleEnv:
    """CartPole-v1 with observations mapped into four finite dimensions."""

    _LOW = np.array([-2.4, -3.0, -0.5, -2.0], dtype=np.float64)
    _HIGH = np.array([2.4, 3.0, 0.5, 2.0], dtype=np.float64)

    def __init__(self, intervals: int = 12):
        if intervals < 2:
            raise ValueError("intervals must be at least 2")
        self._env = gym.make("CartPole-v1")
        self.intervals = intervals
        self.action_space = self._env.action_space
        self.observation_shape = (intervals,) * 4

    def _discretize(self, observation: Sequence[float]) -> State:
        scaled = (np.asarray(observation, dtype=np.float64) - self._LOW) / (self._HIGH - self._LOW)
        bins = np.clip(np.floor(scaled * self.intervals), 0, self.intervals - 1).astype(np.int64)
        return tuple(int(value) for value in bins)  # type: ignore[return-value]

    def reset(self, seed: int | None = None) -> State:
        if seed is None:
            output = self._env.reset()
        else:
            try:
                output = self._env.reset(seed=seed)
            except TypeError:  # legacy gym
                self._env.seed(seed)
                output = self._env.reset()
        observation = output[0] if isinstance(output, tuple) else output
        return self._discretize(observation)

    def step(self, action: int) -> tuple[State, float, bool, bool, dict]:
        output = self._env.step(action)
        if len(output) == 5:
            observation, reward, terminated, truncated, info = output
        else:  # legacy gym
            observation, reward, done, info = output
            terminated, truncated = bool(done), False
        return self._discretize(observation), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._env.close()


class NStepSarsaLearner:
    """Tabular on-policy n-step TD control with an epsilon-greedy policy."""

    def __init__(self, env: DiscreteCartPoleEnv, n_step: int, seed: int = 0):
        if n_step < 1:
            raise ValueError("n_step must be positive")
        self.env = env
        self.n_step = n_step
        self.gamma = 0.99
        self.learning_rate = 0.20
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_episodes = 800
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(env.observation_shape + (env.action_space.n,), dtype=np.float64)

    def epsilon(self, episode: int) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -episode / self.epsilon_decay_episodes
        )

    def greedy_action(self, state: State) -> int:
        values = self.q[state]
        choices = np.flatnonzero(values == values.max())
        return int(self.rng.choice(choices))

    def select_action(self, state: State, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.env.action_space.n))
        return self.greedy_action(state)

    def _update(self, trajectory: list[Transition], start: int) -> None:
        horizon = min(self.n_step, len(trajectory) - start)
        target = sum(self.gamma**offset * trajectory[start + offset][2] for offset in range(horizon))
        last = trajectory[start + horizon - 1]
        if horizon == self.n_step and not last[4] and not last[5]:
            # The action selected at S_{t+n} makes this n-step SARSA.
            target += self.gamma**horizon * float(self.q[last[3]][last[6]])
        state, action = trajectory[start][0:2]
        self.q[state][action] += self.learning_rate * (target - self.q[state][action])

    def train(self, episodes: int, seed_offset: int = 0) -> list[float]:
        episode_returns = []
        for episode in range(episodes):
            state = self.env.reset(seed_offset + episode)
            trajectory: list[Transition] = []
            total_reward = 0.0
            action = self.select_action(state, self.epsilon(episode))
            while True:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = None if terminated or truncated else self.select_action(next_state, self.epsilon(episode))
                trajectory.append((state, action, reward, next_state, terminated, truncated, next_action))
                total_reward += reward
                update_index = len(trajectory) - self.n_step
                if update_index >= 0:
                    self._update(trajectory, update_index)
                if terminated or truncated:
                    # Flush all pending pairs with their available shorter returns.
                    for index in range(max(0, len(trajectory) - self.n_step + 1), len(trajectory)):
                        self._update(trajectory, index)
                    break
                state, action = next_state, next_action
            episode_returns.append(total_reward)
        return episode_returns

    def evaluate(self, episodes: int, seed_offset: int = 10_000) -> list[float]:
        returns = []
        for episode in range(episodes):
            state = self.env.reset(seed_offset + episode)
            total_reward = 0.0
            while True:
                state, reward, terminated, truncated, _ = self.env.step(self.greedy_action(state))
                total_reward += reward
                if terminated or truncated:
                    break
            returns.append(total_reward)
        return returns


def run_experiments(
    n_values: Iterable[int], seeds: Iterable[int], episodes: int, eval_episodes: int,
    intervals: int, output_path: Path,
) -> list[dict[str, float | int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for n_step in n_values:
        for seed in seeds:
            env = DiscreteCartPoleEnv(intervals)
            learner = NStepSarsaLearner(env, n_step, seed)
            training_returns = learner.train(episodes, seed * 100_000)
            evaluation_returns = learner.evaluate(eval_episodes, seed * 100_000 + 50_000)
            env.close()
            row: dict[str, float | int] = {
                "n_step": n_step,
                "seed": seed,
                "episodes": episodes,
                "intervals": intervals,
                "train_last_100_mean": float(np.mean(training_returns[-100:])),
                "eval_mean": float(np.mean(evaluation_returns)),
                "eval_std": float(np.std(evaluation_returns)),
            }
            rows.append(row)
            print(f"n={n_step}, seed={seed}: train={row['train_last_100_mean']:.1f}, eval={row['eval_mean']:.1f} +/- {row['eval_std']:.1f}")
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare n-step SARSA on discretized CartPole.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--intervals", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/nstep_cartpole_results.csv"))
    args = parser.parse_args()
    if min(args.n_values) < 1 or args.episodes < 1 or args.eval_episodes < 1:
        parser.error("n-values, episodes, and eval-episodes must be positive")
    run_experiments(args.n_values, args.seeds, args.episodes, args.eval_episodes, args.intervals, args.output)


if __name__ == "__main__":
    main()
