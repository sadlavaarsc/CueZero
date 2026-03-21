# Performance & Optimization

CueZero achieves state-of-the-art performance through a combination of algorithmic innovations and engineering optimizations. This document details the performance benchmarks and optimization strategies.

---

## Model Efficiency

Despite billiards being a more complex task than traditional board games, CueZero achieves strong performance with an extremely compact model:

| Metric | Value |
|--------|-------|
| **Total parameters** | ~160K |
| **Feature extractor** | 2-layer FC (81→128→128) + GRU |
| **Policy head** | 2-layer FC (128→128→5) |
| **Value head** | 2-layer FC (128→128→1) |

**Why it matters**: The small model size enables:
- Fast inference even on low-end hardware
- Quick training convergence (~1000 epochs total)
- Easy deployment in resource-constrained environments

---

## Competitive Performance

### Win Rates Against Baseline Agents

| Opponent | Win Rate | Rating |
|----------|----------|--------|
| BasicAgent | **95%** | 🏆 Excellent |
| BasicAgentPro | **80%** | 🏆 Excellent |

*Test conditions: 120 games with 4× rotation (first/second turn × ball type distribution)*

The AI demonstrates advanced tactical awareness, including continuous shot capabilities and strategic defense maneuvers. It occasionally achieves "perfect games" where it clears the table in a single turn.

---

## Hardware Configuration

### Training & Evaluation Cluster

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Xeon ICX Platinum 8358 |
| **GPU** | NVIDIA HGX A100 |
| **Cluster** | Siyuan-1 (Shanghai Jiao Tong University) |

### Consumer Hardware (MCTS-Fast)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Optional | NVIDIA RTX 30xx+ |

---

## Performance Benchmarks

### Full Game Latency (60 turns)

| Mode | Hardware | Latency |
|------|----------|---------|
| MCTS-Full | Siyuan-1 Cluster | **< 3 minutes** |
| MCTS-Fast | Consumer PC | **~30 seconds** |

### Decision Latency per Shot

| Mode | Simulations | Depth | Timeout | Latency (Consumer PC) |
|------|-------------|-------|---------|----------------------|
| MCTS-Full | 150 | 4 | 15s | **~3 minutes** |
| MCTS-Fast | 30 | 2 | 3s | **~1 second** |

**Optimization Impact**: MCTS-Fast achieves **180x speedup** over naive full search.

---

## Search Space Reduction Analysis

### The Challenge: Continuous Action Space

The 5-dimensional continuous action space presents an enormous search problem:

| Dimension | Range | Description |
|-----------|-------|-------------|
| V₀ | [0.5, 8.0] | Cue velocity (m/s) |
| φ | [0°, 360°] | Horizontal angle |
| θ | [0°, 90°] | Vertical angle |
| a | [-0.5, 0.5] | Cue offset X |
| b | [-0.5, 0.5] | Cue offset Y |

### Brute-force Search Space

Even with coarse discretization:
- V₀: 15 steps (0.5 increment)
- φ: 36 steps (10° increment)
- θ: 9 steps (10° increment)
- a: 5 steps (0.25 increment)
- b: 5 steps (0.25 increment)

**Total combinations**: 15 × 36 × 9 × 5 × 5 = **121,500**

With fine discretization (for professional play):
- **~243,000+ combinations**

### CueZero's Optimized Search

| Component | Count |
|-----------|-------|
| Heuristic candidates (Ghost Ball method) | ~30 |
| MCTS simulations | 150 |
| **Total evaluations** | **4,500** |

### Search Space Compression

```
Brute-force: 243,000 combinations
CueZero:      4,500 evaluations
──────────────────────────────
Reduction:    54x smaller!
```

---

## Engineering Optimizations

### 1. Heuristic Action Generation

**Ghost Ball Method**: Geometrically calculates optimal shot angles using physics principles

```python
# Generate candidate shots using Ghost Ball heuristic
def generate_heuristic_actions(self, balls, my_targets, table):
    actions = []
    for target_ball in my_targets:
        for pocket in table.pockets:
            # Calculate ghost ball position for straight-in shot
            phi_ideal, distance = self._get_ghost_ball_target(
                cue_pos, obj_pos, pocket_pos
            )
            # Generate variants around ideal shot
            actions.append(...)
    return actions[:30]  # Keep top candidates
```

**Result**: Reduces candidate actions from infinite → ~30 high-quality shots

### 2. Policy-guided Pruning

After generating heuristic candidates, use policy network to filter:

```python
# Keep only top 2/3 candidates closest to policy output
keep_count = max(1, int(self.n_simulations * 2 / 3))
filtered_actions = [action for action, distance in action_distances[:keep_count]]
```

**Result**: Additional 33% reduction in candidates

### 3. Hybrid Evaluation

Combine neural predictions with physical simulation:

```python
# Dynamic weighting between network and simulation
depth_factor = depth / remaining_hits
value = depth_factor * value_output + (1 - depth_factor) * normalized_reward
```

**Benefits**:
- Early depth: More simulation (accurate but slow)
- Late depth: More network (fast but slightly less accurate)

### 4. State Caching & Lightweight Serialization

- Save only essential ball state (position, velocity, pocket status)
- Avoid deep copying entire environment
- Use efficient 81-dimensional state encoding

**Result**: 10-20x faster state saving/restoration

### 5. Simulation Timeouts

Prevent physics simulation hangs:

```python
def simulate_with_timeout(shot, timeout=3):
    signal.alarm(timeout)  # Set timeout
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        return False
```

---

## MCTS Mode Comparison

| Feature | MCTS-Full | MCTS-Fast |
|---------|-----------|-----------|
| **Simulations** | 150 | 30 |
| **Max Depth** | 4 | 2 |
| **Timeout** | 15s | 3s |
| **Candidates** | ~30 | ~10 |
| **Decision Time** | ~3 min (Consumer PC) | ~1 sec (Consumer PC) |
| **Use Case** | Strong play, offline | Real-time, web UI |
| **Win Rate vs Basic** | 95% | 90% |

**Trade-off**: MCTS-Fast sacrifices ~5% win rate for **180x speedup**.

---

## Acknowledgments

The computations in this project were run on the Siyuan-1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University.
