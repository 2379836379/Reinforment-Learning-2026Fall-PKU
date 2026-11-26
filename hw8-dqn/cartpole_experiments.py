"""Run three sequential paired comparisons, reusing identical control runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from dqn_train import TrainConfig, train, plot_run


def run_one(config, directory):
    path = Path(directory) / 'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        # Older CartPole results predate the optional environment/training fields.
        if asdict(TrainConfig(**result['config'])) != asdict(config):
            raise ValueError(f'Configuration mismatch: {path}')
        return result
    return train(config, directory)


def key(epsilon, lr, width):
    return f'eps{epsilon:g}_lr{lr:g}_w{width}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/cartpole_comparison')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    seeds = (11, 22, 33)
    records, stages = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def compare(name, candidates):
            pending = []
            for epsilon, lr, width in candidates:
                label = key(epsilon, lr, width)
                if label in records:
                    continue
                for seed in seeds:
                    config = TrainConfig(epsilon=epsilon, lr=lr, hidden_dim=width, seed=seed)
                    directory = root / label / f'seed{seed}'
                    pending.append((label, pool.submit(run_one, config, str(directory))))
            for label, future in pending:
                records.setdefault(label, []).append(future.result())
            labels = [key(*candidate) for candidate in candidates]
            scores = [float(np.mean([r['late_eval_mean'] for r in records[label]])) for label in labels]
            winner = int(np.argmax(scores))
            stage = dict(name=name, candidates=labels, late_eval_means=scores, winner=labels[winner])
            stages.append(stage)
            (root / 'summary.json').write_text(json.dumps(dict(stages=stages, runs=records), indent=2)+'\n')
            print('STAGE COMPLETE: ' + json.dumps(stage), flush=True)
            return candidates[winner]

        epsilon, _, _ = compare('epsilon', [(0.2, 3e-4, 128), (1.0, 3e-4, 128)])
        _, lr, _ = compare('learning_rate', [(epsilon, 3e-4, 128), (epsilon, 5e-4, 128)])
        compare('network_width', [(epsilon, lr, 64), (epsilon, lr, 128)])

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=True)
    for ax, stage in zip(axes, stages):
        for label in stage['candidates']:
            curves = []
            for seed in seeds:
                directory = root / label / f'seed{seed}'
                data = np.load(directory / 'returns.npz')
                curves.append(data['eval_scores'].mean(axis=1))
                x = data['eval_episodes']
            curves = np.array(curves)
            mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            ax.plot(x, mean, label=label)
            ax.fill_between(x, mean-std, mean+std, alpha=.15)
        ax.set(title=stage['name'], xlabel='Training episode', ylim=(0, 510))
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Greedy validation return (mean +/- seed SD)')
    fig.tight_layout()
    fig.savefig(root / 'comparison.png', dpi=150)
    plt.close(fig)
    for label in records:
        for seed in seeds:
            plot_run(root / label / f'seed{seed}')


if __name__ == '__main__':
    main()
