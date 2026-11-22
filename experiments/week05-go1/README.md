# 第五周：Go1 平地训练与动作缩放实验

讲义：第 10 讲《Go1 平地训练》，第 66–71 页的工程步骤。

代码基线：`bdfe2ed2d70c167538c16b712bd45be8cd023fe3`，MotrixLab `0.4.0b1`。
实验分支：`codex/go1-week05`。原 `main` 保留。

## 本机环境

本机为 Apple M1 Pro，10 核 CPU、32 GiB 统一内存，macOS ARM64。
MotrixSim `0.10.1` 提供 CPython 3.10 的 macOS ARM64 安装包。
仓库的 `motrix_rl/device.py` 按 CUDA、MPS、CPU 的顺序选择训练设备；
因此 NVIDIA 显卡不是运行本实验的必要条件，但讲义的 CUDA 性能配置不能直接用于 Mac。

本实验保留 MotrixSim、RSL-RL、Go1 环境和原网络/奖励设置，采用本机 MPS，
以 128 个并行环境完成 500 次迭代。讲义的 CUDA 示例使用 1024 个环境，本实验明确记录规模差异。

必要组件：Python 3.10、uv、Git LFS、MotrixSim 0.10.1、PyTorch、RSL-RL、TensorBoard。
回放录像使用仓库自带渲染器及 imageio/FFmpeg。

### 安装注意事项

- Mac ARM 上不能安装讲义中的 `tbb==2021.13.0`，本机不启用可选 `tbb` extra。
- 当前锁文件在不启用 CUDA extra 时仍将 RSL-RL 的 torchvision 指向 CUDA wheel。
  本机安装时跳过 torchvision/torchaudio，再单独从 PyPI 安装与实际 Torch 版本匹配的 Mac torchvision。
- 不修改仓库的 `uv.lock`，运行时使用已激活环境或 `.venv/bin/python`，避免裸 `uv run` 触发重装。

```bash
uv sync --frozen --no-default-groups --extra rslrl \
  --no-install-package torchvision --no-install-package torchaudio
uv pip install --python .venv/bin/python --no-deps \
  --index-url https://pypi.org/simple torchvision==0.26.0
uv pip check --python .venv/bin/python
```

实际安装版本：Python 3.10.5、MotrixSim 0.10.1、Torch 2.11.0、torchvision 0.26.0、
RSL-RL 4.0.1、TensorBoard 2.20.0、NumPy 2.2.6。82 个已安装包的兼容检查通过。
512×512 MPS 矩阵计算结果有限；真实 Go1 环境步进后观测和 reward 有限，
Actor 49 维、Critic 52 维、动作 12 维。

Go1 资源按讲义单独下载，无需拉取全部机器人资产：

```bash
git lfs pull \
  --include="motrix_robots/src/motrix_robots/assets/go1/**,motrix_envs/src/motrix_envs/common/motphys-ground.png" \
  --exclude=""
```

## 回放证据采集

`record_follow.py` 复用讲义的跟随相机方法，并从实际环境实例记录 action_scale、
模型 SHA-256、速度、关节位置、接触和终止状态。它不改变网络动作或仿真设置。

先在 `go1.py` 保持 `action_scale=0.1` 运行 A 组，结束进程；仅将该值改为 `0.05`，
用完全相同的命令重新启动 B 组；完成后恢复 `0.1`。

```bash
.venv/bin/python experiments/week05-go1/record_follow.py \
  --policy /absolute/path/to/checkpoints/latest.pt --seconds 10
```

两组使用同一 checkpoint、seed=42、num_envs=1，材料分别保存到
`runs/week05-evidence/A` 和 `runs/week05-evidence/B`。
`--seconds 10` 表示视频时长；真实仿真时长以轨迹步数乘 `ctrl_dt` 计算，单独记录，避免混淆。

也可以使用自动执行和恢复配置的驱动脚本：

```bash
.venv/bin/python experiments/week05-go1/run_ab.py \
  --policy runs/go1-walk-flat/rslrl/torch/ppo/26-11-23_00-48-33-345874/checkpoints/latest.pt
```

## 已完成的真实实验

实验日期：2026-11-23（Asia/Shanghai）。先按讲义执行 16 环境、2 次迭代短测，正常结束并保存模型。
随后执行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/train.py task=go1-walk-flat/rslrl.ppo \
  num_envs=128 seed=42 algo.max_iterations=500 checkpoint.interval=100 \
  logging.backend=tensorboard render=false play=false
```

- Run：`runs/go1-walk-flat/rslrl/torch/ppo/26-11-23_00-48-33-345874`
- 模型：上述目录下 `checkpoints/latest.pt`，4,615,085 bytes。
- 模型 SHA-256：`2c4e6a8f545b02827b1da8f013edf1fd2e48f9086dbe9391030bea28f5a2e10c`
- 训练正常退出（exit 0），完成 500 次 PPO 更新，累计 1,536,000 个环境步。
- 训练循环耗时约 3 分 4 秒，不含安装与启动；最后一次更新约 9,471 环境步/秒。
- 最后平均回报 41.8358，平均回合长度 1000 步。

TensorBoard 实际启动命令：

```bash
.venv/bin/tensorboard --logdir runs/go1-walk-flat --port 6006 --host 127.0.0.1
```

`Loss/value` 有 500 个真实采样点（step 0–499），初值 0.005117、末值 0.000776，
最大值 0.011687。曲线先下降、随后抬升并出现尖峰，约 200 次更新后总体下降，后期仍有偶发波动。
截图关闭了平滑与异常值裁剪。价值损失曲线不能单独证明行走成功，需结合回放。

## A/B 实测结果

两组各从新进程启动，同一模型（SHA-256 相同）、seed=42、num_envs=1、MPS。
均记录 10 秒视频、500 个控制步，真实仿真时长也是 10 秒。
两组首步策略输出的 12 个动作值完全相同。

| 实测项目 | A：action_scale=0.1 | B：action_scale=0.05 |
| --- | ---: | ---: |
| 平均机体前向速度（m/s） | 0.4802 | -0.0122 |
| 前向速度标准差（m/s） | 0.0854 | 0.0352 |
| 最大机体倾角（度） | 7.23 | 2.82 |
| 全段各关节角峰峰值的平均值（rad） | 0.3775 | 0.0866 |
| 摔倒终止 / 超时次数 | 0 / 0 | 0 / 0 |

A 组能持续前进，腿部交替运动，速度接近 0.5 m/s 指令；身体有轻微倾斜，10 秒内没有摔倒。
B 组基本原地小幅运动并略向后移动，关节运动范围和身体摇晃更小，10 秒内同样没有摔倒。

固定默认角与实际执行动作时，B 的目标角偏移是 A 的一半；实测速度却没有减半，说明不能将目标偏移比例
直接套用于闭环运动。实际动作还取决于策略反馈、PD 控制及地面接触。本次只干预同一策略的执行参数，
不能推断 B 组重新训练后的效果。

实验后 `go1.py` 已恢复 `action_scale=0.1`，与基线文件的 Git diff 为空；框架源码未残留临时日志代码。
采集逻辑保存在独立实验脚本中。

## 实验材料

- [环境记录](results/environment.json)
- [原始训练日志](results/train.log)
- [曲线数值摘要](results/training-summary.json)
- [实际 TensorBoard 截图](results/screenshots/tensorboard-value.png)
- [A 组录像](results/A/replay.mp4) / [参数与指标](results/A/summary.json)
- [B 组录像](results/B/replay.mp4) / [参数与指标](results/B/summary.json)
- 两组完整状态轨迹：[A](results/A/trajectory.jsonl) / [B](results/B/trajectory.jsonl)
- [实际任务配置](results/task_config.yaml) / [原始运行元数据](results/metadata.json)
- [第五课飞书答题卡](https://rcnzp4i6r2ht.feishu.cn/wiki/WWCkwxZdyishCnk5wLWcQ5IjnsS)

上述结果已归档到 `results/`，可随实验记录推送。模型权重和周期 checkpoint 仍保存在本机完整运行目录 `runs/`，该目录按仓库惯例被 Git 忽略。
