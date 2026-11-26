"""CartPole / Breakout DQN training with recorded evaluations."""
import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from dqn_agent import DQNAgent
from gym_env import GymEnv, BreakoutEnv
from q_network import QNetwork, BreakoutQNetwork
from sample import FrameNumpy, SampleBatchNumpy


@dataclass
class TrainConfig:
    epsilon: float = 0.20
    lr: float = 5e-4
    hidden_dim: int = 128
    episodes: int = 1000
    seed: int = 11
    gamma: float = 0.99
    buffer_size: int = 10000
    batch_size: int = 64
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    eval_interval: int = 25
    eval_episodes: int = 5
    final_eval_episodes: int = 20
    env: str = 'cartpole'
    device: str = 'cpu'
    learning_starts: int = 0
    train_frequency: int = 1
    epsilon_decay_steps: int = 0
    max_episode_steps: int = 10000


def default_config(env):
    if env == 'breakout':
        return TrainConfig(env=env, epsilon=1.0, epsilon_min=0.1, lr=1e-4,
                           hidden_dim=512, batch_size=32, learning_starts=1000,
                           train_frequency=4, epsilon_decay_steps=100000,
                           eval_interval=50, eval_episodes=3, final_eval_episodes=5)
    if env != 'cartpole':
        raise ValueError(f'Unknown environment: {env}')
    return TrainConfig()


def make_env(config):
    if config.env == 'breakout':
        return BreakoutEnv(max_episode_steps=config.max_episode_steps)
    if config.env == 'cartpole':
        return GymEnv('CartPole-v1')
    raise ValueError(f'Unknown environment: {config.env}')


def evaluate(agent, env, seeds):
    scores = []
    for seed in seeds:
        obs = env.reset({'seed': int(seed)})
        done, total = False, 0.0
        while not done:
            obs, reward, done, _ = env.step(agent.exploit(obs))
            total += reward
        scores.append(total)
    return scores


def train(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    env, eval_env = make_env(config), make_env(config)
    env.reset({'seed': config.seed})
    agent = DQNAgent(dict(action_dim=env.action_dim, epsilon=config.epsilon,
                          gamma=config.gamma, device=config.device))
    if config.env == 'breakout':
        model = BreakoutQNetwork(env.action_dim, config.lr, config.hidden_dim)
    else:
        model = QNetwork(env.state_dim[0], env.action_dim, config.lr, config.hidden_dim)
    agent.set_model(model)
    replay_buffer = deque(maxlen=config.buffer_size)
    train_returns, eval_returns, eval_at = [], [], []
    steps = updates = 0
    started = time.monotonic()
    # Separate, common validation seeds across all configurations and training seeds.
    eval_seeds = range(10000, 10000 + config.eval_episodes)
    try:
        with (output / 'progress.jsonl').open('w') as progress:
            for episode in range(1, config.episodes + 1):
                obs = env.reset()
                done, total = False, 0.0
                while not done:
                    action = agent.predict(obs)
                    next_obs, reward, done, info = env.step(action)
                    action = info.get('executed_action', action)
                    total += reward
                    steps += 1
                    replay_buffer.append(FrameNumpy.from_dict(dict(
                        obs=obs, next_obs=next_obs, action=action, reward=reward, done=done)))
                    obs = next_obs
                    if (len(replay_buffer) > config.batch_size
                            and steps >= config.learning_starts
                            and steps % config.train_frequency == 0):
                        batch = SampleBatchNumpy.stack(random.sample(replay_buffer, config.batch_size))
                        agent.sample_process(batch)
                        agent.learn(batch)
                        updates += 1
                    if config.epsilon_decay_steps:
                        fraction = min(steps / config.epsilon_decay_steps, 1.0)
                        agent.epsilon = config.epsilon + fraction * (config.epsilon_min - config.epsilon)
                train_returns.append(total)
                if not config.epsilon_decay_steps:
                    agent.epsilon = max(config.epsilon_min, agent.epsilon * config.epsilon_decay)
                if episode % config.eval_interval == 0 or episode == config.episodes:
                    scores = evaluate(agent, eval_env, eval_seeds)
                    eval_at.append(episode)
                    eval_returns.append(scores)
                    row = dict(episode=episode, steps=steps, updates=updates, epsilon=agent.epsilon,
                               train_last100=float(np.mean(train_returns[-100:])),
                               eval_mean=float(np.mean(scores)), elapsed_seconds=time.monotonic()-started)
                    progress.write(json.dumps(row) + '\n')
                    progress.flush()
                    if episode % 100 == 0 or episode == config.episodes:
                        print(f'{output.name}: {json.dumps(row)}', flush=True)
        # Held-out initial states, distinct from those used in periodic validation.
        final_scores = evaluate(agent, eval_env, range(20000, 20000 + config.final_eval_episodes))
    finally:
        env.env.close()
        eval_env.env.close()
    result = dict(config=asdict(config), train_last100=float(np.mean(train_returns[-100:])),
                  late_eval_mean=float(np.mean(eval_returns[-5:])),
                  final_eval_mean=float(np.mean(final_scores)), final_eval_scores=final_scores,
                  env_steps=steps, gradient_updates=updates, elapsed_seconds=time.monotonic()-started)
    np.savez_compressed(output / 'returns.npz', train=train_returns, eval_episodes=eval_at,
                        eval_scores=eval_returns, final_scores=final_scores)
    torch.save(dict(config=asdict(config), model=model.state_dict()), output / 'model.pt')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def plot_run(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    output = Path(output)
    data = np.load(output / 'returns.npz')
    config = json.loads((output / 'result.json').read_text())['config']
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(data['train'])+1), data['train'], alpha=.4, label='train')
    ax.plot(data['eval_episodes'], data['eval_scores'].mean(axis=1),
            label=f"greedy validation ({config['eval_episodes']} episodes)")
    title = 'ALE/Breakout-v5' if config.get('env') == 'breakout' else 'CartPole-v1'
    ax.set(xlabel='Training episode', ylabel='Return', title=title)
    ax.legend()
    fig.savefig(output / 'training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=('cartpole', 'breakout'), default='cartpole')
    for name, value in asdict(TrainConfig()).items():
        if name != 'env':
            parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=None)
    parser.add_argument('--output')
    args = parser.parse_args()
    config = default_config(args.env)
    for name in asdict(config):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    positive = ('episodes', 'hidden_dim', 'batch_size', 'train_frequency', 'eval_interval',
                'eval_episodes', 'final_eval_episodes', 'max_episode_steps')
    if any(getattr(config, name) < 1 for name in positive):
        parser.error('episode counts, dimensions, batch size and intervals must be positive')
    if config.buffer_size <= config.batch_size:
        parser.error('buffer-size must exceed batch-size')
    if not 0 <= config.epsilon_min <= config.epsilon <= 1 or config.lr <= 0:
        parser.error('require 0 <= epsilon-min <= epsilon <= 1 and lr > 0')
    if not 0 <= config.gamma <= 1 or not 0 < config.epsilon_decay <= 1:
        parser.error('gamma must be in [0, 1] and epsilon-decay in (0, 1]')
    if config.learning_starts < 0 or config.epsilon_decay_steps < 0:
        parser.error('learning-starts and epsilon-decay-steps must be nonnegative')
    output = args.output or f'results/{args.env}_single'
    print(json.dumps(train(config, output), indent=2))
    plot_run(output)


if __name__ == '__main__':
    main()
