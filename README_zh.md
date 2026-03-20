# CueZero

[English](./README.md) | [中文](./README_zh.md)

![Python](https://img.shields.io/badge/python-3.10-blue)
![Framework](https://img.shields.io/badge/PyTorch-RL-red)
![MCTS](https://img.shields.io/badge/MCTS-Continuous-orange)

CueZero是一个台球强化学习系统，结合了神经网络和连续动作蒙特卡洛树搜索（MCTS）来解决高维决策问题。

## 项目概述

CueZero是一个为连续控制任务设计的高性能台球AI。与传统的用于离散游戏的AlphaZero实现不同，CueZero处理5维连续动作空间和81维状态空间。

该系统集成了：

- 策略-价值神经网络
- 连续动作MCTS
- 基于物理的仿真

以在随机且高度动态的环境中实现强大的性能。

## 关键特性

- **连续动作MCTS**：通过策略引导的局部搜索将传统MCTS扩展到5D连续动作空间
- **策略-价值网络**：从81D状态输入中学习动作分布和状态价值
- **自对弈训练 pipeline**：自动数据生成和迭代模型改进
- **物理仿真集成**：使用pooltool进行精确的环境建模
- **混合评估**：结合学习到的价值估计和基于仿真的滚动

## 架构

![Architecture](./assets/architecture.png)

整个系统遵循神经引导的搜索 pipeline：

CueZero的架构由四个核心组件组成：

1. **策略网络**：接收81维状态向量（包含球的位置、速度和游戏状态）作为输入，并输出5维动作向量（速度、角度等）
2. **价值网络**：与策略网络共享相同的卷积特征提取器，预测当前状态下获胜的概率
3. **蒙特卡洛树搜索**：针对连续动作空间进行优化，使用策略网络引导搜索，价值网络评估位置
4. **闭环训练**：结合专家数据预训练和自对弈，持续改进模型性能

## 训练流程

![training pipeline](./assets/training_pipeline.png)

1. **预训练**：使用BasicAgent生成的数据通过监督学习初始化网络
2. **自对弈训练**：AI与自身对弈生成高质量训练数据
3. **数据集构建**：处理和准备对局数据用于训练
4. **模型训练**：在生成的数据上训练神经网络
5. **评估**：评估模型与基线代理的性能

## 结果

### 基线 Agent

| Agent | 描述 | 水平 |
|-------|------|------|
| **BasicAgent** | 基于贝叶斯优化的规则型 Agent。无明显战术意识，容易犯规和失误。 | 🎱 新手玩家（不熟练） |
| **BasicAgentPro** | 增强物理基线 Agent，具备进攻策略、风险评估和长线规划能力。 | 🎯 熟练玩家（有经验） |

### CueZero 对战基线

| 对手            | 胜率    | 评级     |
|-----------------|---------|----------|
| BasicAgent      | **95%** | 🏆 优秀  |
| BasicAgentPro   | **80%** | 🏆 优秀  |

*测试条件：120 局比赛，4 的倍数轮换（先后手×球型分配）*

AI展示了先进的战术意识，包括连续击球能力和战略防守 maneuver。它偶尔会出现"完美游戏"，在一个回合内清台。

### 性能基准

- **≥70% vs BasicAgent**: 优秀性能阈值
- **CueZero 达到 95%**: 显著超越基线
- **80% vs BasicAgentPro**: 对抗增强物理基线的强劲表现
## 核心挑战与解决方案

| 挑战                                | 解决方案                                                               |
| ----------------------------------- | ---------------------------------------------------------------------- |
| 高维度（81D状态，5D动作）           | 带有卷积特征提取的自定义神经网络架构                                   |
| 连续动作空间                        | 带有局部邻域搜索的策略引导MCTS                                         |
| 强不确定性                          | 结合神经预测和物理仿真的混合评估                                       |
| 实时要求                            | 带有剪枝和有限节点扩展的优化MCTS                                       |
| 混沌行为                            | 幽灵球启发式 + 随机扰动采样 + 策略筛选                                 |

## 快速开始

### 环境配置

```bash
# 创建 conda 环境（推荐 Python 3.10+）
# 注意：本项目使用 'poolenv' 环境
conda activate poolenv

# 如果还没有环境，创建它：
conda create -n poolenv python=3.10
conda activate poolenv

# 安装 pooltool（台球物理引擎）
# 从源码安装（推荐）：
git clone https://github.com/bboland/pooltool.git
cd pooltool && pip install -e .
cd ..

# 安装依赖
pip install -r requirements.txt

# 安装包（可选，用于开发）
pip install -e .
```

### 运行评估

```bash
# 对抗基线代理评估
python scripts/evaluate.py
```

### Web UI

CueZero 包含一个基于 Web 的 UI，用于可视化和与 AI 交互。

```bash
# 启动 Web 服务器（mock 模式 - 默认）
cd server
python server.py

# 使用 AI 模式（需要 dual_network_final.pt 模型文件）
python server.py --ai

# 访问 http://localhost:8000
```

Web UI 支持：
- 可视化台球桌和球的位置
- 执行击球并查看结果
- 获取 AI 推荐的击球动作（AI 模式）

### CLI 对战

CueZero 提供命令行界面进行 Agent 对战。

```bash
# 快速开始：MCTS(Fast) vs Basic
python scripts/cli_game.py --agent-a mcts_fast --agent-b basic --games 5

# Human vs MCTS(Full)
python scripts/cli_game.py --agent-a human --agent-b mcts_full --games 3

# 查看完整参数
python scripts/cli_game.py --help
```

**参数说明**：
- `--agent-a`: Agent A 类型 (human, mcts_fast, mcts_full, policy, basic, random)
- `--agent-b`: Agent B 类型
- `--games`: 对战局数
- `--model`: 模型文件路径
- `--no-verbose`: 禁用详细输出
- `--seed`: 随机种子

### Agent 类型说明

| 类型         | 描述                                   | 使用场景           |
|--------------|----------------------------------------|--------------------|
| human        | 人工操作，通过命令行输入               | 人机对战、测试     |
| mcts_fast    | MCTS 快速模式 (30 次模拟，深度 2)        | 实时对战、快速测试 |
| mcts_full    | MCTS 完整模式 (150 次模拟，深度 4)       | 强对弈、离线分析   |
| policy       | 策略网络直接预测                       | 快速推理           |
| basic        | 基于启发式规则                         | 基线对比           |
| random       | 随机动作                               | 测试、调试         |

### 服务端对战 API

服务器支持通过 REST API 进行 Agent 对战。

```bash
# 启动服务器（指定默认 Agent）
python server.py --agent-a mcts_fast --agent-b basic

# 使用 API 开始对战
curl -X POST http://localhost:8000/api/battle/start \
  -H "Content-Type: application/json" \
  -d '{"agent_a_type": "human", "agent_b_type": "mcts_fast", "total_games": 3}'

# 执行下一步（自动获取 AI 决策）
curl -X POST http://localhost:8000/api/battle/{battle_id}/next \
  -H "Content-Type: application/json" \
  -d '{}'

# 获取对战状态
curl http://localhost:8000/api/battle/{battle_id}/status

# 列出所有对战
curl http://localhost:8000/api/battles
```

### MCTS 模式

MCTS 实现支持两种模式，用于不同的使用场景：

```python
from cuezero.mcts.search import MCTS

# 快速模式：快速决策（30 次模拟，深度 2, 3 秒超时）
mcts_fast = MCTS(model=model, mode="fast")

# 完整模式：强对弈（150 次模拟，深度 4, 15 秒超时）
mcts_full = MCTS(model=model, mode="full")

# 自定义配置（覆盖模式默认值）
mcts_custom = MCTS(model=model, mode="fast", n_simulations=50, max_depth=3)
```

| 模式   | 模拟次数 | 最大深度 | 超时   | 使用场景         |
|--------|----------|----------|--------|------------------|
| fast   | 30       | 2        | 3s     | 实时对弈、Web UI |
| full   | 150      | 4        | 15s    | 离线分析、强对弈 |

### 训练

```bash
# 生成自对弈数据
python scripts/selfplay.py

# 训练模型
python scripts/train.py
```

### 配置

修改 `configs/` 目录中的 YAML 文件：
- `model.yaml`: 网络架构和输入/输出维度
- `training.yaml`: 训练超参数
- `mcts.yaml`: MCTS 搜索参数

## 模型文件

预训练模型文件 `dual_network_final.pt` 应放置在项目根目录。

模型架构包括：
- **SharedFeatureExtractor**: 处理 3x81 维状态向量（连续 3 局状态）
- **PolicyHead**: 输出 5 维动作向量（速度、角度、偏移）
- **ValueHead**: 输出胜率估计

模型文件格式：PyTorch state dict，包含 `feature_extractor`、`policy_head` 和 `value_head` 组件。

## 项目结构

```
cuezero/
│
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── configs/
│   ├── training.yaml
│   ├── mcts.yaml
│   └── model.yaml
│
├── cuezero/
│   ├── env/
│   │   ├── billiards_env.py     # 台球环境实现
│   │   ├── state_encoder.py     # 神经网络状态编码
│   │   └── physics_wrapper.py   # 物理引擎包装器
│   │
│   ├── models/
│   │   ├── policy_network.py    # 动作预测策略网络
│   │   ├── value_network.py     # 位置评估价值网络
│   │   └── networks.py          # 组合策略-价值网络
│   │
│   ├── mcts/
│   │   ├── tree.py              # MCTS树实现
│   │   ├── node.py              # MCTS节点实现
│   │   └── search.py            # MCTS搜索算法
│   │
│   ├── selfplay/
│   │   ├── selfplay_worker.py   # 自对弈游戏生成
│   │   └── dataset_builder.py   # 训练数据集构建
│   │
│   ├── training/
│   │   ├── trainer.py           # 模型训练逻辑
│   │   ├── loss.py              # 损失函数
│   │   └── replay_buffer.py     # 经验回放缓冲区
│   │
│   ├── inference/
│   │   └── agent.py             # 推理代理
│   │
│   └── utils/
│       ├── logger.py            # 日志工具
│       └── config.py            # 配置管理
│
├── scripts/
│   ├── train.py                 # 训练脚本
│   ├── selfplay.py              # 自对弈数据生成
│   ├── evaluate.py              # 评估脚本
│   └── cli_game.py              # CLI 对战界面
│
├── server/
│   ├── server.py                # Web 服务器（含对战 API）
│   └── mock_env.py              # Mock 环境
│
└── experiments/
    └── baseline_eval.py         # 基线代理评估
```

## 未来方向

### 系统与工程

- **模块化服务架构**：将训练、自对弈和推理重构为具有清晰接口的独立服务（如RPC/REST），实现可扩展部署
- **分布式自对弈**：跨多个工作节点或机器并行化数据生成，提高训练吞吐量
- **实验管理**：集成配置跟踪、日志记录和可复现性工具（如结构化配置、版本化检查点）
- **性能优化**：通过批处理、缓存和混合精度推理优化MCTS和仿真瓶颈
- **模型服务与部署**：通过推理API公开训练好的代理，用于实时决策或外部集成
- **可视化与调试工具**：构建轨迹回放、决策检查和训练诊断工具

### 算法与建模

- **改进策略表示**：探索更具表达力的模型（如注意力机制）以实现更好的空间推理
- **高效连续搜索**：研究连续动作空间中更样本高效的探索策略
- **不确定性建模**：将随机建模或置信度估计纳入价值预测
- **课程学习**：设计渐进式训练方案，稳定复杂环境中的学习
- **混合方法**：将基于学习的方法与分析或基于物理的先验相结合

## 致谢

该项目最初是在学术环境中开发的，后来被重构为独立的工程项目。实现灵感来自AlphaZero原理，并适应了台球的独特挑战。