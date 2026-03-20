"""
CueZero Agent Module

统一 Agent 接口，支持多种 Agent 类型：
- MCTSAgent: 基于 MCTS 搜索的 Agent
- PolicyAgent: 基于策略网络的 Agent
- HumanAgent: 人工操作 Agent
- BasicAgent: 基于规则/贝叶斯优化的 Agent
- RandomAgent: 随机动作 Agent
"""

import collections
import math
import random
import numpy as np
import torch


class Agent:
    """Base agent class for CueZero"""

    def __init__(self, name="Agent"):
        self.name = name

    def decision(self, balls, my_targets, table) -> dict:
        """Make a decision and return action as dictionary.

        Args:
            balls: Ball state dictionary {ball_id: Ball object}
            my_targets: List of target ball IDs for current player
            table: Table object for dimensions

        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        raise NotImplementedError

    def reset(self):
        """Reset agent state"""
        pass


class MCTSAgent(Agent):
    """MCTS agent for billiards"""

    def __init__(self, mcts, env=None, name="MCTSAgent"):
        super().__init__(name=name)
        self.mcts = mcts
        self.env = env
        self.state_buffer = collections.deque(maxlen=3)
        self.hit_count = 0  # Hit counter for tracking game progress

    def clear_buffer(self):
        """Clear state buffer and reset hit counter"""
        self.state_buffer.clear()
        self.hit_count = 0

    def reset(self):
        """Reset agent state"""
        self.clear_buffer()

    def set_env(self, env):
        """Set environment (needed for MCTS search)"""
        self.env = env

    def decision(self, balls, my_targets=None, table=None):
        """Use MCTS to make a decision.

        Args:
            balls: Ball state dictionary
            my_targets: Target ball ID list
            table: Table object

        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # Encode current state to 81-dimensional vector
        state_vec = self.mcts._balls_state_to_81(balls, my_targets, table)

        # Maintain last 3 states buffer
        if len(self.state_buffer) < 3:
            self.state_buffer.append(state_vec)
            while len(self.state_buffer) < 3:
                self.state_buffer.append(state_vec)
        else:
            self.state_buffer.append(state_vec)

        state_seq = list(self.state_buffer)

        # Calculate remaining hits (max 60 in a game)
        remaining_hits = 60 - self.hit_count

        # Get player info from env or default
        if self.env is not None:
            curr_player = self.env.get_curr_player()
            target_ball = getattr(self.env, 'target_ball', 'solid')
        else:
            curr_player = 'A'
            target_ball = 'solid'

        # Build player_targets dict from env
        if target_ball == 'solid':
            player_targets = {
                'A': ['1', '2', '3', '4', '5', '6', '7'],
                'B': ['9', '10', '11', '12', '13', '14', '15']
            }
        else:
            player_targets = {
                'A': ['9', '10', '11', '12', '13', '14', '15'],
                'B': ['1', '2', '3', '4', '5', '6', '7']
            }

        # MCTS search for best action
        action = self.mcts.search(
            state_seq, balls, table,
            player_targets, curr_player,
            remaining_hits
        )

        # Increment hit counter
        self.hit_count += 1

        # Convert numpy array to dict format
        return {
            "V0": float(action[0]),
            "phi": float(action[1]),
            "theta": float(action[2]),
            "a": float(action[3]),
            "b": float(action[4]),
        }


class PolicyAgent(Agent):
    """Policy network agent for billiards (direct prediction without MCTS)"""

    def __init__(self, policy_network, device="cuda" if torch.cuda.is_available() else "cpu", name="PolicyAgent"):
        super().__init__(name=name)
        self.policy_network = policy_network
        self.device = device
        self.state_buffer = collections.deque(maxlen=3)

        # Action normalization bounds (consistent with MCTS)
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)

    def clear_buffer(self):
        """Clear state buffer"""
        self.state_buffer.clear()

    def reset(self):
        """Reset agent state"""
        self.clear_buffer()

    def _denormalize_action(self, action_norm):
        """Denormalize action from [0,1] to physical range"""
        action_norm = np.clip(action_norm, 0.0, 1.0)
        return action_norm * (self.action_max - self.action_min) + self.action_min

    def _balls_state_to_81(self, balls, my_targets, table):
        """Convert ball state to 81-dimensional vector"""
        state = np.zeros(81, dtype=np.float32)

        ball_order = [
            'cue', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15'
        ]

        STANDARD_TABLE_WIDTH = 2.845
        STANDARD_TABLE_LENGTH = 1.4225
        BALL_RADIUS = 0.0285

        for i, ball_id in enumerate(ball_order):
            base = i * 4

            if ball_id in balls:
                ball = balls[ball_id]
                rvw = ball.state.rvw
                pos = rvw[0]

                if ball.state.s == 4:  # Pocketed
                    state[base + 0] = -1.0
                    state[base + 1] = -1.0
                    state[base + 2] = -1.0
                    state[base + 3] = 1.0
                else:
                    state[base + 0] = pos[0] / STANDARD_TABLE_WIDTH
                    state[base + 1] = pos[1] / STANDARD_TABLE_LENGTH
                    state[base + 2] = pos[2] / (2 * BALL_RADIUS)
                    state[base + 3] = 0.0
            else:
                state[base + 0] = -1.0
                state[base + 1] = -1.0
                state[base + 2] = -1.0
                state[base + 3] = 1.0

        # Table dimensions
        if table is not None:
            state[64] = table.w
            state[65] = table.l
        else:
            state[64] = 2.540
            state[65] = 1.270

        # Target ball one-hot encoding
        if my_targets:
            for t in my_targets:
                idx = int(t) - 1
                if 0 <= idx <= 14:
                    state[66 + idx] = 1.0

        return state

    def decision(self, balls, my_targets=None, table=None):
        """Direct policy network prediction without MCTS.

        Args:
            balls: Ball state dictionary
            my_targets: Target ball ID list
            table: Table object

        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # Encode current state
        state_vec = self._balls_state_to_81(balls, my_targets, table)

        # Maintain last 3 states buffer
        if len(self.state_buffer) < 3:
            self.state_buffer.append(state_vec)
            while len(self.state_buffer) < 3:
                self.state_buffer.append(state_vec)
        else:
            self.state_buffer.append(state_vec)

        state_seq = list(self.state_buffer)

        # Convert to model input format
        states = np.stack(state_seq, axis=0)
        # Apply state preprocessing (normalization)
        from cuezero.env.state_encoder import StateEncoder
        preprocessor = StateEncoder()
        states = preprocessor.process_three_game_states(states)

        state_tensor = torch.from_numpy(states).float().unsqueeze(0).to(self.device)

        # Direct model prediction
        with torch.no_grad():
            out = self.policy_network(state_tensor)
            action_norm = out["policy_output"][0].cpu().numpy()

        # Denormalize to physical action
        action = self._denormalize_action(action_norm)

        return {
            "V0": float(action[0]),
            "phi": float(action[1]),
            "theta": float(action[2]),
            "a": float(action[3]),
            "b": float(action[4]),
        }


class HumanAgent(Agent):
    """Human player agent - gets actions from command line input"""

    def __init__(self, name="HumanAgent"):
        super().__init__(name=name)

    def decision(self, balls, my_targets=None, table=None):
        """Get action from human player via command line input.

        Args:
            balls: Ball state dictionary
            my_targets: Target ball ID list
            table: Table object

        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        print("\n" + "="*50)
        print("轮到你了！请输入击球参数：")
        print(f"目标球：{my_targets}")
        print("="*50)

        while True:
            try:
                # Get V0
                v0 = float(input("V0 (力度，0.5-8.0 m/s): ").strip())
                if not 0.5 <= v0 <= 8.0:
                    print("力度超出范围，请重新输入 (0.5-8.0)")
                    continue

                # Get phi
                phi = float(input("phi (水平角度，0-360 度): ").strip())
                phi = phi % 360  # Normalize to [0, 360)

                # Get theta
                theta = float(input("theta (垂直角度，0-90 度): ").strip())
                if not 0 <= theta <= 90:
                    print("垂直角度超出范围，请重新输入 (0-90)")
                    continue

                # Get a
                a = float(input("a (横向偏移，-0.5 到 0.5): ").strip())
                if not -0.5 <= a <= 0.5:
                    print("横向偏移超出范围，请重新输入 (-0.5 到 0.5)")
                    continue

                # Get b
                b = float(input("b (纵向偏移，-0.5 到 0.5): ").strip())
                if not -0.5 <= b <= 0.5:
                    print("纵向偏移超出范围，请重新输入 (-0.5 到 0.5)")
                    continue

                action = {
                    "V0": v0,
                    "phi": phi,
                    "theta": theta,
                    "a": a,
                    "b": b,
                }

                print(f"\n确认动作：V0={v0:.2f}, phi={phi:.2f}, theta={theta:.2f}, a={a:.3f}, b={b:.3f}")
                confirm = input("确认？(y/n): ").strip().lower()
                if confirm == 'y':
                    return action
                else:
                    print("请重新输入\n")

            except ValueError:
                print("输入无效，请输入数字")
            except KeyboardInterrupt:
                print("\n\n输入中断，使用随机动作")
                return self._random_action()

    def _random_action(self):
        """Generate random action as fallback"""
        return {
            "V0": round(random.uniform(0.5, 8.0), 2),
            "phi": round(random.uniform(0, 360), 2),
            "theta": round(random.uniform(0, 90), 2),
            "a": round(random.uniform(-0.5, 0.5), 3),
            "b": round(random.uniform(-0.5, 0.5), 3),
        }


class BasicAgent(Agent):
    """Rule-based / Bayesian optimization agent for billiards.

    This is a simplified version adapted from the original AI3603-Billiards project.
    Uses Bayesian optimization to search for optimal shot parameters.
    """

    def __init__(self, name="BasicAgent"):
        super().__init__(name=name)

        # Search space bounds
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }

        # Optimization parameters
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2

    def _random_action(self):
        """Generate random action"""
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }

    def decision(self, balls=None, my_targets=None, table=None):
        """Use Bayesian optimization to find best shot.

        For now, returns a heuristic-based action.
        Full Bayesian optimization implementation can be added later.
        """
        if balls is None:
            print(f"[BasicAgent] No balls received, using random action.")
            return self._random_action()

        # Simple heuristic: aim at nearest target ball
        try:
            import pooltool as pt

            cue_ball = balls.get('cue')
            if cue_ball is None:
                return self._random_action()

            cue_pos = cue_ball.state.rvw[0][:2]

            # Find nearest target ball
            best_target = None
            best_dist = float('inf')

            for tid in my_targets:
                if tid in balls and balls[tid].state.s != 4:
                    target_pos = balls[tid].state.rvw[0][:2]
                    dist = np.linalg.norm(np.array(target_pos) - np.array(cue_pos))
                    if dist < best_dist:
                        best_dist = dist
                        best_target = tid

            if best_target is None:
                # No valid target, try to hit 8-ball
                if '8' in balls and balls['8'].state.s != 4:
                    best_target = '8'
                else:
                    return self._random_action()

            # Calculate aim direction
            target_pos = balls[best_target].state.rvw[0][:2]
            dx = target_pos[0] - cue_pos[0]
            dy = target_pos[1] - cue_pos[1]
            phi = math.degrees(math.atan2(dy, dx))
            if phi < 0:
                phi += 360

            # Estimate power based on distance
            v0 = min(2.0 + best_dist * 0.5, 5.0)

            action = {
                'V0': round(v0, 2),
                'phi': round(phi, 2),
                'theta': 45.0,
                'a': 0.0,
                'b': 0.0,
            }

            print(f"[BasicAgent] Aiming at ball {best_target}: V0={action['V0']:.2f}, phi={action['phi']:.2f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] Decision error: {e}, using random action")
            return self._random_action()


class RandomAgent(Agent):
    """Random action agent for testing"""

    def __init__(self, name="RandomAgent"):
        super().__init__(name=name)

    def decision(self, balls=None, my_targets=None, table=None):
        """Return random action.

        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
        return action
