"""CartPole training with reproducible configuration and recorded evaluations."""
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
from gym_env import GymEnv
from q_network import QNetwork
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


def evaluate(agent, env, seeds):
    scores = []
    for seed in seeds:
        obs, _ = env.env.reset(seed=int(seed))
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
    env, eval_env = GymEnv('CartPole-v1'), GymEnv('CartPole-v1')
    env.env.reset(seed=config.seed)
    agent = DQNAgent(dict(action_dim=env.action_dim, epsilon=config.epsilon,
                          gamma=config.gamma, device='cpu'))
    model = QNetwork(env.state_dim[0], env.action_dim, config.lr, config.hidden_dim)
    agent.set_model(model)
    replay_buffer = deque(maxlen=config.buffer_size)
    train_returns, eval_returns, eval_at = [], [], []
    steps = 0
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
                    next_obs, reward, done, _ = env.step(action)
                    total += reward
                    steps += 1
                    replay_buffer.append(FrameNumpy.from_dict(dict(
                        obs=obs, next_obs=next_obs, action=action, reward=reward, done=done)))
                    obs = next_obs
                    if len(replay_buffer) > config.batch_size:
                        batch = SampleBatchNumpy.stack(random.sample(replay_buffer, config.batch_size))
                        agent.sample_process(batch)
                        agent.learn(batch)
                train_returns.append(total)
                agent.epsilon = max(config.epsilon_min, agent.epsilon * config.epsilon_decay)
                if episode % config.eval_interval == 0 or episode == config.episodes:
                    scores = evaluate(agent, eval_env, eval_seeds)
                    eval_at.append(episode)
                    eval_returns.append(scores)
                    row = dict(episode=episode, steps=steps, epsilon=agent.epsilon,
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
                  env_steps=steps, elapsed_seconds=time.monotonic()-started)
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
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(data['train'])+1), data['train'], alpha=.4, label='train')
    ax.plot(data['eval_episodes'], data['eval_scores'].mean(axis=1), label='greedy validation (5 episodes)')
    ax.set(xlabel='Training episode', ylabel='Return', title='CartPole-v1')
    ax.legend()
    fig.savefig(output / 'training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for name in ('epsilon', 'lr', 'hidden_dim', 'episodes', 'seed'):
        value = getattr(defaults, name)
        parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    parser.add_argument('--output', default='results/cartpole_single')
    args = parser.parse_args()
    config = TrainConfig(**{name: getattr(args, name) for name in ('epsilon', 'lr', 'hidden_dim', 'episodes', 'seed')})
    if config.episodes < 1 or config.hidden_dim < 1 or not 0 <= config.epsilon <= 1 or config.lr <= 0:
        parser.error('episodes, hidden-dim and lr must be positive; epsilon must be in [0, 1]')
    print(json.dumps(train(config, args.output), indent=2))
    plot_run(args.output)


if __name__ == '__main__':
    main()
