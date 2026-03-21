# CueZero: High-Performance Billiards AI System

[English](./README.md) | [中文](./README_zh.md)

![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![MCTS](https://img.shields.io/badge/MCTS-Continuous-orange)
![License](https://img.shields.io/badge/license-MIT-green)

![Demo](./assets/demo.gif)

## 📌 Project Overview

CueZero is a high-performance billiards AI system that combines deep reinforcement learning with a specially engineered continuous-action Monte Carlo Tree Search (MCTS). It solves the challenging problem of decision-making in a high-dimensional continuous state and action space with complex physics dynamics.

**Key Highlights**:
- 81-dimensional state representation, 5-dimensional continuous action space
- **Compact model (~160K parameters)** despite billiards being more complex than traditional board games
- Specialized MCTS achieving **54x search space reduction**
- **95% win rate** against rule-based baseline agents
- 60-turn game in **< 3 minutes** on Siyuan-1 cluster
- MCTS-Fast mode: **180x speedup** (3 minutes → 1 second) for consumer hardware

---

## 🎯 Key Results

### Competitive Performance

| Opponent | Win Rate | Rating |
|----------|----------|--------|
| BasicAgent | **95%** | 🏆 Excellent |
| BasicAgentPro | **80%** | 🏆 Excellent |

*Test conditions: 120 games with 4× rotation (first/second turn × ball type distribution)*

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Full game (60 turns)** | < 3 minutes (Siyuan-1 cluster) |
| **MCTS-Full decision** | ~3 minutes per shot (Consumer PC) |
| **MCTS-Fast decision** | ~1 second per shot (Consumer PC) |
| **Search space reduction** | 54x smaller than brute force |
| **Model size** | ~160K parameters (very compact!) |
| **Training efficiency** | Basic proficiency in ~200 epochs |
| **Total training** | ~1000 epochs |

### Hardware Configuration

**Training Cluster (Siyuan-1)**:
- CPU: Intel Xeon ICX Platinum 8358
- GPU: NVIDIA HGX A100

**Consumer Deployment**:
- MCTS-Fast runs on standard laptops/desktops

---

## 🚀 Quick Start

### Prerequisites
- Ubuntu 22.04 (recommended)
- Python 3.13
- Conda

### Minimal Installation

```bash
# Create and activate conda environment
conda create -n poolenv python=3.13
conda activate poolenv

# Install pooltool physics engine
git clone https://github.com/SJTU-RL2/pooltool.git
cd pooltool
pip install "poetry==2.2.1"
poetry install --with=dev,docs
cd ..

# Install CueZero dependencies
pip install -r requirements.txt
pip install bayesian-optimization numpy
```

See [docs/INSTALLATION.md](./docs/INSTALLATION.md) for detailed installation guide.

### Run Your First Game

```bash
# CLI: MCTS-Fast vs BasicAgent (5 games)
python scripts/cli_game.py --agent-a mcts_fast --agent-b basic --games 5

# Web UI (mock mode, no model required)
PYTHONPATH=. python -m server.server

# Web UI with AI (requires dual_network_final.pt)
PYTHONPATH=. python -m server.server --ai
```

Access the Web UI at: http://localhost:8000

---

## 📂 Architecture

CueZero's architecture follows a neural-guided search pipeline:

```
Game State (81D)
      │
      ▼
┌─────────────────────────────────┐
│  Policy-Value Network           │
│  - Shared Feature Extractor     │
│  - Policy Head (5D action)     │
│  - Value Head (win prob)       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Continuous-Action MCTS         │
│  - Ghost Ball Heuristic         │
│  - Policy-guided Pruning        │
│  - Hybrid Evaluation            │
└─────────────┬───────────────────┘
              │
              ▼
      Optimal Shot (5D)
```

### Core Components

1. **Policy-Value Network**: Takes 3 consecutive 81D states, outputs 5D action distribution and win probability
2. **Continuous-Action MCTS**: Specialized for billiards with heuristic search and policy guidance
3. **Self-Play Pipeline**: Automated data generation and iterative model improvement
4. **Physics Simulation**: Integrates with pooltool for accurate environment modeling

See [docs/HOW_IT_WORKS.md](./docs/HOW_IT_WORKS.md) for deep technical dive.

---

## 🔧 Engineering Highlights

### 1. Compact Model Architecture for Complex Task

**The Challenge**: Billiards is inherently more complex than traditional board games like Chess/Go, with continuous physics, stochastic outcomes, and high-dimensional state/action spaces.

**The Solution**: A carefully designed lightweight network:
- **Shared Feature Extractor**: 2-layer FC + GRU for spatio-temporal fusion
- **Policy Head**: 2-layer FC for 5D action prediction
- **Value Head**: 2-layer FC for win probability estimation
- **Total parameters**: Only **~160K** (extremely compact!)

**Why it works**: Smart architecture design prioritizes essential features (ball positions, velocities, pocket status) while avoiding unnecessary complexity. The model achieves strong performance despite its small size.

### 2. Specialized MCTS for Continuous Action Space

**The Challenge**: 5D continuous action space with ~243,000+ potential combinations (even with coarse discretization).

**The Solution**:
- **Ghost Ball heuristic**: Geometrically generates ~30 high-quality candidates
- **Policy-guided pruning**: Keeps top 2/3 candidates (66% reduction)
- **Result**: 54x smaller search space (4,500 evaluations vs 243,000 combinations)

```
Brute-force: 243,000 combinations
CueZero:      4,500 evaluations
──────────────────────────────
Reduction:    54x smaller!
```

### 3. Dual MCTS Modes for Different Use Cases

| Feature | MCTS-Full | MCTS-Fast |
|---------|-----------|-----------|
| Simulations | 150 | 30 |
| Max Depth | 4 | 2 |
| Timeout | 15s | 3s |
| Decision Time | ~3 min (Consumer PC) | ~1 sec (Consumer PC) |
| Win Rate vs Basic | 95% | 90% |
| Use Case | Strong play | Real-time, web UI |

**MCTS-Fast**: 180x speedup with only 5% win rate trade-off.

### 4. Efficient Training Pipeline

- **Pre-training**: ~200 epochs on BasicAgent data (learns basic shot-making)
- **Self-play Training**: ~600 epochs with MCTS-guided data generation
- **Supplementary Training**: ~200 epochs for specialized refinement
- **Total**: ~1000 epochs to reach full performance

**Key Optimization**: Heuristic search accelerates training by 3-5x compared to naive RL.

See [docs/TRAINING.md](./docs/TRAINING.md) for complete training documentation.

### 5. Hybrid Evaluation Strategy

Combines neural network predictions with physical simulation:
- **Early depth**: More simulation (accurate but slow)
- **Late depth**: More network (fast but slightly less accurate)
- **Dynamic weighting**: Smooth transition based on search depth

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [docs/INSTALLATION.md](./docs/INSTALLATION.md) | Detailed installation and environment setup guide |
| [docs/TRAINING.md](./docs/TRAINING.md) | Complete training pipeline and self-play documentation |
| [docs/PERFORMANCE.md](./docs/PERFORMANCE.md) | Performance benchmarks and optimization details |
| [docs/HOW_IT_WORKS.md](./docs/HOW_IT_WORKS.md) | Deep technical dive into architecture and algorithms |

---

## 🎮 Usage Examples

### CLI Battle

```bash
# Human vs MCTS-Full
python scripts/cli_game.py --agent-a human --agent-b mcts_full --games 3

# MCTS-Fast vs BasicAgentPro
python scripts/cli_game.py --agent-a mcts_fast --agent-b basic_pro --games 10

# View all options
python scripts/cli_game.py --help
```

### Web UI

```bash
# Start with default agents
PYTHONPATH=. python -m server.server --agent-a mcts_fast --agent-b basic
```

### REST API

```bash
# Start new battle
curl -X POST http://localhost:8000/api/battle/start \
  -H "Content-Type: application/json" \
  -d '{"agent_a_type": "human", "agent_b_type": "mcts_fast", "total_games": 3}'

# Execute next step
curl -X POST http://localhost:8000/api/battle/{battle_id}/next \
  -H "Content-Type: application/json" -d '{}'
```

### Agent Types

| Type | Description | Use Case |
|------|-------------|----------|
| `human` | Human player via CLI/Web UI | Human vs AI |
| `mcts_fast` | Fast MCTS (30 sims, depth 2, 3s) | Real-time play, web UI |
| `mcts_full` | Full MCTS (150 sims, depth 4, 15s) | Strong play, offline |
| `policy` | Policy network direct output | Fast inference |
| `basic` | Heuristic rule-based | Baseline comparison |
| `basic_pro` | Enhanced physics-based | Advanced baseline |
| `random` | Random actions | Testing, debugging |

---

## 🙏 Acknowledgments

The computations in this project were run on the **Siyuan-1 cluster** supported by the Center for High Performance Computing at Shanghai Jiao Tong University.

This project was initially developed as a course project at Shanghai Jiao Tong University, and later refined into a standalone engineering project. The implementation draws inspiration from AlphaZero principles adapted to the unique challenges of continuous-action billiards.

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🔗 Related

- [pooltool](https://github.com/SJTU-RL2/pooltool) - Billiards physics engine
- [AlphaZero](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) - Inspiration for this project
