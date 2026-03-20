import numpy as np
import torch
import collections
from .tree import Tree


class Agent:
    """Base agent class for CueZero"""

    def __init__(self):
        pass

    def decision(self, balls, my_targets, table):
        """Make a decision and return action as dictionary.

        Args:
            balls: Ball state dictionary {ball_id: Ball object}
            my_targets: List of target ball IDs for current player
            table: Table object for dimensions

        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        raise NotImplementedError


class MCTSAgent(Agent):
    """MCTS agent for billiards"""

    def __init__(self, mcts, env):
        super().__init__()
        self.mcts = mcts
        self.env = env
        self.state_buffer = collections.deque(maxlen=3)
        self.hit_count = 0  # Hit counter for tracking game progress

    def clear_buffer(self):
        """Clear state buffer and reset hit counter"""
        self.state_buffer.clear()
        self.hit_count = 0

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

        # Get player info from env
        player_targets = self.env.player_targets
        curr_player = self.env.get_curr_player()

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

    def __init__(self, policy_network, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.policy_network = policy_network
        self.device = device
        self.state_buffer = collections.deque(maxlen=3)

        # Action normalization bounds (consistent with MCTS)
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)

    def clear_buffer(self):
        """Clear state buffer"""
        self.state_buffer.clear()

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
