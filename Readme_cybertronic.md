# Cybertronic 运行指南 - VBot 第一赛段导航任务

本文档说明了如何在原始 MotrixLab 例程基础上进行核心修改，以解决机器人无法在目标区域停稳以及起始位置不准确的问题，使代码能够成功运行并稳定收敛。

## 1. 核心修复逻辑 (Core Fixes)

### 1.1 起始位置对齐 (Radial Spawn)
原始代码通常使用固定点偏移，导致机器人出生在非指定区域。
- **修改位置**: `vbot_section001_np.py` 中的 `reset()` 方法。
- **实现方案**: 
    - 采集采样半径 $R = 5.4m$（精准对应场景中白色箭头圆周）。
    - 随机采样角度 $\theta \in [0, 2\pi]$，计算起始坐标 $X = R \cos\theta, Y = R \sin\theta$。
    - **自动定向**: 设置初始 Yaw 角为 $\theta + \pi$，确保机器人出生即面向圆心。

### 1.2 三层停止保障机制 (Stopping Mechanism)
单纯依靠奖励函数难以在高速（1.5m/s）下实现精确刹车，必须引入物理层干预：
1. **动作衰减 (Action Damping)**: 在 `apply_action` 中，距离目标 2.0m 开始线性衰减策略输出，0.3m 处动作完全归零。
2. **物理阻尼自适应 (Adaptive KV)**: 在 `_compute_torques` 中，机器人接近目标时，PD 控制器的速度增益 $k_v$ 从 2.0 增大到 **10.0**，提供强力物理制动。
3. **粘性惩罚 (Ever Reached)**: 引入 `ever_reached` 状态位，一旦机器狗进入圆圈后再次跳出，给予 **-30** 的重罚。

## 2. 环境配置调整 (cfg.py)

在 `VBotSection001EnvCfg` 中进行以下关键对齐：
- **坐标归零**: `InitState.pos = [0.0, 0.0, 0.5]`，将坐标系原点设为圆盘中心。
- **半径配置**: 新增 `spawn_radius = 5.4`。
- **目标配置**: `Commands.pose_command_range` 设为全零，对齐 `reset` 中的原点目标逻辑。

## 3. 运行步骤

### 3.1 训练环境拉取
同步代码至最新版本：
```bash
git pull
```

### 3.2 启动分布式训练
推荐使用 2048 个环境以获得最佳梯度效果：
```bash
python scripts/train.py --env=vbot_navigation_section001 --num-envs=2048
```

### 3.3 结果回放与验证
使用 `play.py` 查看机器狗是否能准确起始于箭头处并在圆心稳稳停住：
```bash
python scripts/play.py --env=vbot_navigation_section001
```

## 4. 关键参数参考

| 参数 | 推荐值 | 作用 |
|------|-------|------|
| `damping_radius` | 2.0m | 刹车缓冲区长度 |
| `stop_radius` | 0.3m | 物理停止阈值 |
| `kv_stop` | 10.0 | 停止状态下的 D 项增益 |
| `max_episode_steps` | 4000 | 保证有足够时间完成长距离导航 |

---
*Documentation by Cybertronic Team for MotrixLab Navigation Task.*
