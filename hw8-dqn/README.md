# DQN：CartPole 与 Breakout

安装依赖后运行：

```bash
uv sync
uv run python dqn_train.py --env cartpole
uv run python dqn_train.py --env breakout --n-step 3 --output results/breakout_3step
```

Breakout 使用 Gymnasium 的 `ALE/Breakout-v5`；当前项目环境中的 `ale-py` 已包含可用的 Breakout ROM。程序默认使用 CPU，有可用的 CUDA 环境时可加 `--device cuda:0`。默认 CartPole 参数继续采用已有实验选出的设置。

## Breakout 输入与网络

每步重复动作 4 个游戏帧，不使用 sticky actions。画面转灰度、裁剪并下采样为 80×80，再堆叠最近四帧，得到 `(4, 80, 80)` 输入。经验池存储 uint8 图像，网络推理时才转浮点并除以 255。

Q 网络为三个卷积层（32 通道 8×8/步长 4、64 通道 4×4/步长 2、64 通道 3×3/步长 1），接 Dueling 价值与优势双分支，每个分支有 256 单元隐藏层，按 `Q = V + A - mean(A)` 合成动作 Q 值。可用 `--no-dueling --hidden-dim 512` 切回原单分支网络。开局和失去生命后自动发球，经验记录实际执行的 FIRE 动作；评估采用同样的封装。失去一条生命不会结束回合，游戏结束或达到步数上限才结束。Breakout 的时间上限截断会结束采样回合，但 TD 目标仍保留下一状态价值。

## Breakout 默认训练配置

- 学习率 `5e-5`，gamma `0.99`，batch size `32`，经验池容量 `10000`。
- 收集 `10000` 个环境步后开始更新，此后每 `4` 个环境步更新一次。
- epsilon 在 `250000` 个环境步内从 `1.0` 线性降到 `0.1`，随后保持；此模式不使用按回合的 `epsilon_decay`。
- 训练 `1000` 回合，每回合最多 `10000` 个环境步。这里的环境步为一次动作决策，通常相当于 4 个游戏帧。
- 目标网络每 `1000` 次梯度更新硬同步一次（更新开始后通常相隔 4000 个环境步）；使用 Double DQN、Huber 损失和全局梯度范数上限 `10`。
- 使用 `3-step` 回报：`R = r0 + gamma*r1 + gamma²*r2`，后接 `gamma³` 折扣的 Double DQN 下一状态价值。回合末尾不足三步时按实际长度累计；真实终止取消 bootstrap，Breakout 时间截断保留 bootstrap。队列在回合结束时排空，不跨回合拼接。`--n-step 1` 可恢复一步更新；CartPole 默认仍为 1-step。
- 每 `50` 回合评估 `10` 局，结束后另测 `20` 局；评估 epsilon 为 `0.01`，使用每局独立且固定种子的随机数生成器，不影响训练 RNG。奖励使用原始游戏分数。

此前 1-step 优化版的 1000/3000 回合完整训练已完成，延长训练未带来持续改善；本次新增 3-step 尚未完成完整训练，不能确认收益。1000 回合不保证覆盖 250000 步探索衰减，应结合 `progress.jsonl` 中的步数和 epsilon 判断训练预算。完整分析与实际验证见 CHANGELOG。

与已有 3000 回合、seed=11 的 1-step 结果作对照：

```bash
uv run python dqn_train.py --env breakout --n-step 3 --episodes 3000 \
  --seed 11 --output results/breakout_3step_3000_seed11
```

该命令仅改变回报步数，网络和其他训练参数沿用当前配置。回合预算一致不保证交互步数一致，比较时同时报告实际步数。3-step 使用采样行为策略的中间奖励，未加入重要性采样修正；可能加快奖励传播，也可能增加方差及离策略偏差。训练/评估报告的分数仍是原始未折扣回合奖励。旧 checkpoint 缺失 `n_step` 时按 1-step 读取。

可通过 `--lr`、`--batch-size`、`--buffer-size`、`--learning-starts`、`--train-frequency`、`--epsilon-decay-steps`、`--target-update-freq` 等调整；`--hidden-dim` 控制每个分支的隐藏层宽度。`--no-double-dqn` 保留目标网络但使用目标网络最大 Q 值；`--no-double-dqn --target-update-freq 0` 关闭目标网络；`--no-huber-loss --max-grad-norm 0` 恢复 MSE 且不裁剪梯度。完整参数见 `--help`。

输出目录保存 `model.pt`、`best_model.pt`、`result.json`、`progress.jsonl`、`returns.npz` 和 `training.png`。`best_model.pt` 按验证均分选择（同分保留更早模型）；最终测试针对最后的 `model.pt`，不代表最佳模型得分。进度增加各验证区间更新的平均 loss、Q 值、TD 目标和裁剪前梯度范数。两个模型文件均包含网络权重与训练配置，不包含恢复训练所需的完整状态。已有运行目录会拒绝覆盖，请使用新目录。

以统一协议重测旧模型或新模型（默认 CPU、20 局、种子 20000–20019、epsilon=0.01）：

```bash
uv run python evaluate_checkpoint.py results/breakout_train/model.pt \
  --output results/breakout_baseline_reevaluation.json
uv run python evaluate_checkpoint.py results/breakout_optimized/model.pt \
  --output results/breakout_optimized_reevaluation.json
```

脚本从保存配置重建网络，兼容原单分支 CNN；可用 `--epsilon 0` 复现贪心评估。新旧成绩应使用相同评估协议比较。

快速验证 Breakout 采样、更新、评估和保存流程：

```bash
uv run python dqn_train.py --env breakout --episodes 3 \
  --max-episode-steps 120 --learning-starts 32 --buffer-size 256 \
  --batch-size 8 --target-update-freq 10 --eval-interval 1 \
  --eval-episodes 2 --final-eval-episodes 3 --output results/breakout_smoke_new
uv run python -m unittest test_n_step test_dqn test_breakout -v
```

短测试仅验证运行流程，不代表学会游戏。CartPole 默认仍采用先前实验的网络、超参数和原算法开关；本轮算法优化默认仅对 Breakout 开启。
