"""Compare Breakout checkpoints, untrained networks and simple policies."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from dqn_agent import DQNAgent
from dqn_train import TrainConfig
from gym_env import BreakoutEnv
from q_network import BreakoutQNetwork


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config = TrainConfig(**saved['config'])
    if config.env != 'breakout':
        raise ValueError(f'Expected a Breakout checkpoint: {path}')
    return saved, config


def make_agent(config, weights=None, init_seed=None):
    if init_seed is not None:
        torch.manual_seed(init_seed)
    model = BreakoutQNetwork(4, config.lr, config.hidden_dim, config.dueling)
    if weights is not None:
        model.load_state_dict(weights)
    model.eval()
    agent = DQNAgent(dict(action_dim=4, device='cpu'))
    agent.set_model(model)
    return agent


def score_policy(env, policy, seeds, epsilon):
    """policy=None is uniform random; callable policies receive obs.

    Matches dqn_train.evaluate's epsilon RNG for non-random policies. All
    policies use the same environment reset seeds and automatic FIRE wrapper.
    """
    scores, lengths = [], []
    for seed in seeds:
        obs = env.reset({'seed': int(seed)})
        rng = np.random.default_rng(int(seed))
        done, total, steps = False, 0., 0
        while not done:
            if policy is None or rng.random() < epsilon:
                action = int(rng.integers(env.action_dim))
            else:
                action = policy(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
            steps += 1
        scores.append(total)
        lengths.append(steps)
    return dict(scores=scores, decision_steps=lengths, mean=float(np.mean(scores)),
                median=float(np.median(scores)), std_population=float(np.std(scores)),
                min=float(np.min(scores)), max=float(np.max(scores)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True,
                        help='Checkpoint defining the architecture of untrained networks')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--init-seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=30000)
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--max-episode-steps', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.episodes < 1 or args.seed_start < 0 or args.max_episode_steps < 1
            or not 0 <= args.epsilon <= 1 or any(s < 0 for s in args.init_seeds)):
        parser.error('invalid episode count, seeds, epsilon or time limit')
    if len(set(args.init_seeds)) != len(args.init_seeds):
        parser.error('init-seeds must be unique')
    if args.output.exists():
        parser.error('output directory exists; choose a new directory')
    torch.set_num_threads(1)
    _, template = load_checkpoint(args.template)
    policies = []
    for entry in args.checkpoint:
        label, separator, path = entry.partition('=')
        if not label or not separator or not path:
            parser.error('checkpoint must be LABEL=PATH')
        saved, config = load_checkpoint(path)
        agent = make_agent(config, weights=saved['model'])
        metadata = dict(kind='checkpoint', path=path, config=asdict(config),
                        saved_episode=saved.get('episode'), saved_eval_mean=saved.get('eval_mean'))
        policies.append((label, agent.exploit, metadata))
    for seed in args.init_seeds:
        agent = make_agent(template, init_seed=seed)
        policies.append((f'untrained_seed{seed}', agent.exploit,
                         dict(kind='untrained', init_seed=seed, template=str(args.template),
                              hidden_dim=template.hidden_dim, dueling=template.dueling)))
    policies.append(('random', None, dict(kind='uniform_random')))
    env = BreakoutEnv(max_episode_steps=args.max_episode_steps)
    try:
        meanings = env.env.unwrapped.get_action_meanings()
        for name in ('NOOP', 'LEFT', 'RIGHT'):
            action = meanings.index(name)
            policies.append((f'fixed_{name.lower()}', lambda obs, a=action: a,
                             dict(kind='fixed', action_name=name, action=action)))
        labels = [label for label, _, _ in policies]
        if len(labels) != len(set(labels)):
            parser.error('policy labels must be unique, including generated baselines')
        args.output.mkdir(parents=True)
        protocol = dict(episodes=args.episodes, seed_start=args.seed_start,
                        eval_epsilon=args.epsilon, max_episode_steps=args.max_episode_steps,
                        env='ALE/Breakout-v5', frameskip=4, sticky_action_probability=0.,
                        automatic_fire=True, random_start=False,
                        note='epsilon perturbation applies to checkpoints, untrained and fixed policies; '
                             'random policy is uniformly random before automatic FIRE')
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        results = []
        for index, (label, policy, metadata) in enumerate(policies):
            print(f'Evaluating {label} ({index + 1}/{len(policies)})', flush=True)
            result = dict(label=label, **metadata,
                          **score_policy(env, policy, range(args.seed_start, args.seed_start + args.episodes),
                                         args.epsilon))
            results.append(result)
            # Numerical filenames allow arbitrary labels without path traversal.
            (args.output / f'policy_{index:02d}.json').write_text(json.dumps(result, indent=2) + '\n')
            print(f'{label}: mean={result["mean"]:.3f}, median={result["median"]:.3f}, '
                  f'std={result["std_population"]:.3f}', flush=True)
        (args.output / 'summary.json').write_text(json.dumps(dict(protocol=protocol, policies=results), indent=2) + '\n')
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            fields = ['label', 'kind', 'mean', 'median', 'std_population', 'min', 'max']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
    finally:
        env.env.close()


if __name__ == '__main__':
    main()
