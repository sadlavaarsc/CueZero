import numpy as np


class StateEncoder:
    """
    Encode billiards game state into 81-dimensional vector for neural network input.

    State vector breakdown:
    - 16 balls × 4 features = 64 dimensions (position xyz + pocketed flag)
    - 2 table dimensions (width, length)
    - 15 target ball flags (one-hot encoding for balls 1-15)
    Total: 81 dimensions
    """

    def __init__(self, table_width=2.845, table_length=1.4225):
        """
        Initialize state encoder.

        Args:
            table_width: Standard table width in meters
            table_length: Standard table length in meters
        """
        self.table_width = table_width
        self.table_length = table_length
        self.ball_radius = 0.0285  # Standard billiard ball radius in meters

        # Ball ordering for consistent encoding
        self.ball_order = [
            'cue', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15'
        ]

    def encode(self, balls, my_targets=None, table=None):
        """
        Encode raw ball state into 81-dimensional vector.

        Args:
            balls: Ball state dictionary {ball_id: Ball object}
            my_targets: List of target ball IDs for current player
            table: Table object for dimensions

        Returns:
            np.ndarray: 81-dimensional state vector
        """
        state = np.zeros(81, dtype=np.float32)

        # Encode each ball's state (64 dimensions for 16 balls × 4 features)
        for i, ball_id in enumerate(self.ball_order):
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
                    # Position normalized by table dimensions
                    state[base + 0] = pos[0] / self.table_width
                    state[base + 1] = pos[1] / self.table_length
                    state[base + 2] = pos[2] / (2 * self.ball_radius)
                    state[base + 3] = 0.0
            else:
                # Ball not in game - mark as pocketed
                state[base + 0] = -1.0
                state[base + 1] = -1.0
                state[base + 2] = -1.0
                state[base + 3] = 1.0

        # Encode table dimensions (indices 64-65)
        if table is not None:
            state[64] = table.w
            state[65] = table.l
        else:
            # Default table dimensions
            state[64] = self.table_width
            state[65] = self.table_length

        # Encode target ball flags (indices 66-80, 15 dimensions for balls 1-15)
        if my_targets:
            for t in my_targets:
                if t.isdigit():
                    idx = int(t) - 1
                    if 0 <= idx <= 14:
                        state[66 + idx] = 1.0

        return state

    def normalize_state(self, state):
        """
        Normalize an 81-dimensional state vector.

        Args:
            state: 81-dimensional state vector

        Returns:
            Normalized state vector
        """
        state = state.copy()

        # Ball coordinates normalization (dimensions 0-63)
        # x coordinate (0, 4, 8, ..., 60) normalized to 0-1
        for i in range(0, 64, 4):
            if state[i] != -1:  # Not pocketed
                state[i] = state[i] / self.table_width

        # y coordinate (1, 5, 9, ..., 61) normalized to 0-1
        for i in range(1, 64, 4):
            if state[i] != -1:  # Not pocketed
                state[i] = state[i] / self.table_length

        # z coordinate (2, 6, 10, ..., 62) normalized by ball diameter
        for i in range(2, 64, 4):
            if state[i] != -1:  # Not pocketed
                state[i] = state[i] / (2 * self.ball_radius)

        # Table dimensions normalization (64-65)
        state[64] = state[64] / self.table_width
        state[65] = state[65] / self.table_length

        # Target ball flags (66-80) remain unchanged (binary values)

        return state

    def process_three_game_states(self, three_game_states):
        """
        Process continuous three-game state vectors.

        Args:
            three_game_states: Array of shape [3, 81] for consecutive game states

        Returns:
            Processed 3D state array with normalized values
        """
        if three_game_states.shape != (3, 81):
            raise ValueError(f"Expected shape (3, 81), got {three_game_states.shape}")

        processed_states = np.zeros_like(three_game_states)
        for i in range(3):
            processed_states[i] = self.normalize_state(three_game_states[i])

        return processed_states

    def get_input_shape(self):
        """Get input shape for neural network."""
        return (81,)

    def __call__(self, states):
        """Support direct calling for processing states."""
        return self.process_three_game_states(states)
