"""Evaluate saved weights under a common protocol, including older checkpoints."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from dqn_agent import DQNAgent
from dqn_train import TrainConfig, evaluate, make_env
from q_network import BreakoutQNetwork, QNetwork


def evaluate_checkpoint(checkpoint, episodes=20, seed_start=20000, epsilon=.01):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = TrainConfig(**saved['config'])
    config.device = 'cpu'
    env = make_env(config)
    try:
        if config.env == 'breakout':
            model = BreakoutQNetwork(env.action_dim, config.lr, config.hidden_dim, config.dueling)
        else:
            model = QNetwork(env.state_dim[0], env.action_dim, config.lr, config.hidden_dim)
        model.load_state_dict(saved['model'])
        model.eval()
        # A target network is unnecessary for inference.
        agent = DQNAgent(dict(action_dim=env.action_dim, device='cpu'))
        agent.set_model(model)
        scores = evaluate(agent, env, range(seed_start, seed_start + episodes), epsilon)
    finally:
        env.env.close()
    return dict(checkpoint=str(checkpoint), config=asdict(config), episodes=episodes,
                seed_start=seed_start, eval_epsilon=epsilon, scores=scores,
                mean=float(np.mean(scores)), std=float(np.std(scores)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=20000)
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or not 0 <= args.epsilon <= 1:
        parser.error('require episodes > 0, seed-start >= 0 and epsilon in [0, 1]')
    if args.output.exists():
        parser.error('output exists; choose a new path')
    result = evaluate_checkpoint(args.checkpoint, args.episodes, args.seed_start, args.epsilon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
