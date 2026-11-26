# DQN：CartPole 与 Breakout

安装依赖后运行：

```bash
uv sync
uv run python dqn_train.py --env cartpole
uv run python dqn_train.py --env breakout --output results/breakout_train
```

Breakout 使用 Gymnasium 的 `ALE/Breakout-v5`；当前项目环境中的 `ale-py` 已包含可用的 Breakout ROM。程序默认使用 CPU，有可用的 CUDA 环境时可加 `--device cuda:0`。默认 CartPole 参数继续采用已有实验选出的设置。

## Breakout 输入与网络

每步重复动作 4 个游戏帧，不使用 sticky actions。画面转灰度、裁剪并下采样为 80×80，再堆叠最近四帧，得到 `(4, 80, 80)` 输入。经验池存储 uint8 图像，网络推理时才转浮点并除以 255。

Q 网络为三个卷积层（32 通道 8×8/步长 4、64 通道 4×4/步长 2、64 通道 3×3/步长 1），接 512 单元全连接层，输出各动作 Q 值。开局和失去生命后自动发球，经验记录实际执行的 FIRE 动作；评估采用同样的封装。失去一条生命不会结束回合，游戏结束或达到步数上限才结束。

## Breakout 默认训练配置

- 学习率 `1e-4`，gamma `0.99`，batch size `32`，经验池容量 `10000`。
- 收集 `1000` 个环境步后开始更新，此后每 `4` 个环境步更新一次。
- epsilon 在 `100000` 个环境步内从 `1.0` 线性降到 `0.1`，随后保持；此模式不使用按回合的 `epsilon_decay`。
- 训练 `1000` 回合，每回合最多 `10000` 个环境步。这里的环境步为一次动作决策，通常相当于 4 个游戏帧。
- 每 `50` 回合贪心评估 `3` 局，结束后另测 `5` 局。奖励使用原始游戏分数。

这些参数是可运行的起点，尚未通过长时间实验优化。可通过 `--lr`、`--batch-size`、`--buffer-size`、`--learning-starts`、`--train-frequency`、`--epsilon-decay-steps`、`--max-episode-steps` 等参数调整；`--hidden-dim` 控制卷积网络后的全连接层宽度。完整参数见 `--help`。

输出目录保存 `model.pt`、`result.json`、`progress.jsonl`、`returns.npz` 和 `training.png`。`model.pt` 包含网络权重与训练配置，不包含优化器、经验池或恢复训练的完整状态。

快速验证 Breakout 采样、更新、评估和保存流程：

```bash
uv run python dqn_train.py --env breakout --episodes 2 \
  --max-episode-steps 100 --learning-starts 32 --buffer-size 256 \
  --batch-size 8 --eval-interval 1 --eval-episodes 1 --final-eval-episodes 1 \
  --output results/breakout_smoke
uv run python -m unittest test_breakout -v
```

短测试仅验证运行流程，不代表学会游戏。DQN 更新公式保持原实现，尚未加入目标网络等算法改进；时间上限截断仍沿用原框架的 `done` 处理方式。
