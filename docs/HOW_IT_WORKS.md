# How It Works

This document explains the technical details of how CueZero works under the hood. We'll dive into the architecture, algorithms, and key innovations that make this system effective.

---

## System Architecture Overview

CueZero follows a neural-guided search architecture inspired by AlphaZero, but specially adapted for continuous action spaces and the unique challenges of billiards.

![Architecture](../assets/architecture.png)

---

## State Representation: 81-Dimensional Vector

The game state is encoded as an 81-dimensional vector that captures all essential information about the current situation.

### State Breakdown

| Range | Content | Description |
|-------|---------|-------------|
| 0-63 | Ball states | 16 balls × 4 features each |
| 64-65 | Table dimensions | Table width and length |
| 66-80 | Target balls | One-hot encoding of current player's targets |

### Ball State Features (each ball)

| Index | Feature | Description |
|-------|---------|-------------|
| +0 | X position | Normalized by table width |
| +1 | Y position | Normalized by table length |
| +2 | Z position | Normalized by ball diameter |
| +3 | Status | 0 = on table, 1 = pocketed |

### Ball Order
```
['cue', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
```

### Why 3 Consecutive States?

The network receives **3 consecutive states** (3 × 81D = 243D total) as input:
- Captures motion information (velocities implicitly)
- Provides temporal context
- Helps the network understand shot outcomes

---

## Action Representation: 5-Dimensional Vector

Each shot is defined by 5 continuous physical parameters:

| Dimension | Name | Range | Description |
|-----------|------|-------|-------------|
| 0 | V₀ | [0.5, 8.0] | Cue velocity (m/s) - controls shot power |
| 1 | φ | [0°, 360°] | Horizontal aim angle |
| 2 | θ | [0°, 90°] | Vertical cue elevation (0° = level, 90° = masse) |
| 3 | a | [-0.5, 0.5] | Cue offset X (side spin) |
| 4 | b | [-0.5, 0.5] | Cue offset Y (follow/draw) |

---

## Policy-Value Network

### Architecture

```
Input (3 × 81D state sequence)
    │
    ▼
┌─────────────────────────────────┐
│  Shared Feature Extractor       │
│  - Conv1D layers                │
│  - Batch Normalization          │
│  - ReLU activation              │
└─────────────┬───────────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌───────────┐   ┌───────────┐
│ Policy    │   │ Value     │
│ Head      │   │ Head      │
│           │   │           │
│ 5D action │   │ Scalar    │
│ [0,1]     │   │ [0,1]     │
└───────────┘   └───────────┘
```

### Outputs

1. **Policy Output**: 5-dimensional vector in [0,1] normalized space
   - Denormalized to physical ranges using action_min/action_max
   - Provides prior for MCTS search

2. **Value Output**: Scalar in [0,1] representing win probability
   - 0 = certain loss
   - 1 = certain win
   - Used for fast state evaluation in MCTS

---

## Continuous-Action MCTS

Traditional MCTS is designed for discrete action spaces. CueZero uses a specially adapted version for continuous billiards actions.

### Key Innovations

#### 1. Heuristic Action Generation with Ghost Ball

Instead of sampling random actions, we generate geometrically-informed candidates:

```python
def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
    """Calculate ideal shot using Ghost Ball heuristic."""
    # Vector from object ball to pocket
    vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
    dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)

    # Unit vector pointing from object ball to pocket
    unit_vec = vec_obj_to_pocket / dist_obj_to_pocket

    # Ghost ball position: 2 ball radii behind object ball
    ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)

    # Calculate angle from cue to ghost ball
    vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
    phi = self._calc_angle_degrees(vec_cue_to_ghost)

    return phi, dist_cue_to_ghost
```

For each target ball and pocket combination:
1. Calculate ghost ball position
2. Generate ideal shot
3. Create variants with slight perturbations

**Result**: ~30 high-quality candidates instead of random sampling

#### 2. Policy-guided Pruning

After generating heuristic candidates, we use the policy network to focus on the most promising ones:

```python
# Calculate distance between each candidate and policy output
for action in candidate_actions:
    phi_diff = abs(action_arr[1] - model_action[1])
    if phi_diff > 180:
        phi_diff = 360 - phi_diff

    v0_diff = abs(action_arr[0] - model_action[0])
    distance = (phi_diff / 180.0) * 0.7 + (v0_diff / 7.5) * 0.3
    action_distances.append((action, distance))

# Sort by distance and keep top 2/3
action_distances.sort(key=lambda x: x[1])
keep_count = max(1, int(self.n_simulations * 2 / 3))
filtered_actions = [action for action, distance in action_distances[:keep_count]]
```

#### 3. MCTS Loop

The core search loop:

```python
for simulation in range(n_simulations):
    # 1. Selection: Pick action using UCB
    if np.sum(N) < n_candidates:
        idx = int(np.sum(N))  # Round-robin for first pass
    else:
        ucb_values = Q + c_puct * sqrt(log(sum(N) + 1) / (N + eps))
        idx = argmax(ucb_values)

    # 2. Simulation: Execute with noise
    shot = simulate_action(balls, table, action)

    # 3. Evaluation: Hybrid of network and simulation
    normalized_reward = compute_reward(shot)
    value = depth_factor * value_output + (1 - depth_factor) * normalized_reward

    # 4. Backpropagation: Update statistics
    N[idx] += 1
    Q[idx] += (value - Q[idx]) / N[idx]

# Select action with highest average reward
best_idx = argmax(Q)
```

#### 4. Hybrid Value Estimation

We combine neural predictions with physics simulation results:

```python
# Dynamic weighting based on depth
depth_factor = depth / remaining_hits
value = depth_factor * value_output + (1 - depth_factor) * normalized_reward
```

**Why this works**:
- **Shallow depth**: Physics simulation is accurate and not too slow
- **Deep depth**: Network prediction is fast and avoids compounding simulation errors

---

## Reward Function

The reward function analyzes shot outcomes to guide learning:

| Outcome | Reward |
|---------|--------|
| Scratch + 8-ball in | -500 |
| 8-ball in prematurely | -500 |
| Scratch (cue in) | -100 |
| Foul: wrong first hit | -30 |
| Foul: no cushion hit | -30 |
| Enemy ball potted | -20 each |
| Neutral shot (no foul, no balls) | +10 |
| Own ball potted | +50 each |
| 8-ball legally potted (win) | +150 |

**Normalization**: Rewards are scaled to [0, 1] for MCTS:
```python
normalized_reward = (raw_reward - (-500)) / 650.0
normalized_reward = clip(normalized_reward, 0.0, 1.0)
```

---

## Training Pipeline

The overall system follows a neural-guided search pipeline with iterative self-improvement:

![Training Pipeline](../assets/training_pipeline.png)

See [TRAINING.md](./TRAINING.md) for detailed training documentation.

---

## Summary of Key Innovations

1. **Continuous-action MCTS**: Adapts MCTS to 5D continuous action space
2. **Ghost Ball heuristic**: Generates high-quality shot candidates geometrically
3. **Policy-guided pruning**: Focuses search on promising actions
4. **Hybrid evaluation**: Combines network speed with simulation accuracy
5. **State caching**: Efficient state representation and restoration
6. **Closed-loop training**: Iterative self-improvement with MCTS guidance

These innovations enable CueZero to achieve strong performance while remaining computationally tractable.
