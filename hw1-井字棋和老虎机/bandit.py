"""Epsilon-greedy experiments for the normal-distribution bandit."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class NormalDistBandit:
    def __init__(self, means, stds):
        if len(means) != len(stds):
            raise ValueError("Means and stds must be the same length.")
        self.n = len(means)
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)
        if np.any(self.stds < 0):
            raise ValueError("Stds must be non-negative.")

    def pull(self, k):
        if not 0 <= k < self.n:
            raise ValueError(f"Invalid arm {k}.")
        return np.random.normal(loc=self.means[k], scale=self.stds[k])


def epsilon_greedy(values, epsilon):
    """Choose a greedy arm with probability 1-epsilon.

    Exploration is sampled from the non-greedy arms, so epsilon is exactly
    the probability of exploring rather than an approximate probability that
    also occasionally re-selects the greedy arm.
    """
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("There should be 2 or more values.")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    greedy = int(np.argmax(values))
    if np.random.random() >= epsilon:
        return greedy
    alternatives = np.delete(np.arange(len(values)), greedy)
    return int(np.random.choice(alternatives))


def run_experiment(bandit, epsilon, iterations):
    """Return the running average reward for one independent experiment."""
    values = np.zeros(bandit.n, dtype=float)
    counts = np.zeros(bandit.n, dtype=np.int64)
    average_reward = np.zeros(iterations, dtype=float)
    total = 0.0

    for i in range(iterations):
        action = epsilon_greedy(values, epsilon)
        counts[action] += 1
        reward = bandit.pull(action)
        values[action] += (reward - values[action]) / counts[action]
        total += reward
        average_reward[i] = total / (i + 1)
    return average_reward


def main():
    parser = argparse.ArgumentParser(description="Plot epsilon-greedy bandit curves.")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--output", default="bandit_epsilon_comparison.png")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    n = 5
    epsilons = [0.01, 0.05, 0.1, 0.2]
    bandit = NormalDistBandit(means=np.arange(-n, n + 1), stds=np.ones(2 * n + 1))
    x = np.arange(args.iterations)

    plt.figure(figsize=(8, 5))
    for epsilon in epsilons:
        # Reset the stream so curves differ mainly because of epsilon.
        np.random.seed(args.seed)
        rewards = run_experiment(bandit, epsilon, args.iterations)
        plt.plot(x, rewards, label=f"epsilon={epsilon:g}")

    plt.axhline(np.max(bandit.means), color="black", linestyle="--",
                linewidth=1, label="best arm mean")
    plt.xlabel("Iterations")
    plt.ylabel("Average reward")
    plt.title("Epsilon-greedy bandit comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
