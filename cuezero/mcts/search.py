import math
import numpy as np
import copy
import torch
import signal
import time


class SimulationTimeoutError(Exception):
    """Physical simulation timeout exception"""
    pass


def _timeout_handler(signum, frame):
    """Timeout signal handler"""
    raise SimulationTimeoutError("Physical simulation timeout")


def simulate_with_timeout(shot, timeout=3):
    """Physical simulation with timeout protection"""
    # Set timeout signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # Set timeout

    try:
        import pooltool as pt
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # Cancel timeout
        return True
    except SimulationTimeoutError:
        return False
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # Restore original handler


class MCTSNode:
    """MCTS Node for billiards"""

    def __init__(self, state_seq, parent=None, prior=1.0, depth=0):
        self.state_seq = state_seq
        self.parent = parent
        self.children = {}
        self.depth = depth

        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Average value
        self.P = prior  # Prior probability


class MCTS:
    """Multi-step MCTS for continuous action billiards"""

    def __init__(self,
                 model,
                 n_simulations=150,
                 c_puct=1.414,
                 max_depth=4,
                 max_search_time=15.0,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize MCTS.

        Args:
            model: Policy-value neural network
            n_simulations: Number of MCTS simulations
            c_puct: UCB exploration constant
            max_depth: Maximum search depth
            max_search_time: Maximum search time in seconds
            device: torch device for model inference
        """
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.max_search_time = max_search_time
        self.device = device
        self.ball_radius = 0.028575

        # Noise levels for simulation
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

        # Action bounds
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)

        # State preprocessor
        from cuezero.env.state_encoder import StateEncoder
        self.state_preprocessor = StateEncoder()

        print("Multi-step MCTS initialized.")

    def _calc_angle_degrees(self, v):
        """Calculate angle in degrees from vector"""
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """Calculate ghost ball target for heuristic shot generation.

        Args:
            cue_pos: Cue ball position
            obj_pos: Object ball position
            pocket_pos: Target pocket position

        Returns:
            tuple: (phi angle in degrees, distance to ghost ball)
        """
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)

        if dist_obj_to_pocket == 0:
            return 0, 0

        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)

        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)

        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table, only_eight_ball=False):
        """Generate candidate action list.

        Args:
            balls: Ball state dictionary
            my_targets: Target ball list for current player
            table: Table object
            only_eight_ball: If True, only generate 8-ball shots

        Returns:
            list: List of action dictionaries
        """
        actions = []

        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]

        cue_pos = cue_ball.state.rvw[0]

        # Get target ball IDs
        if only_eight_ball:
            target_ids = ['8'] if '8' in balls and balls['8'].state.s != 4 else []
        else:
            target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]

        # No target balls left (fallback to 8-ball)
        if not target_ids:
            target_ids = ['8']

        # Generate shots for each target ball
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # Generate shots for each pocket
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # Calculate ideal shot angle
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # Estimate power based on distance
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # Generate variants
                actions.append({'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0})
                actions.append({'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0})
                actions.append({'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0})
                actions.append({'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0})

        # Fallback to random actions if no heuristic shots generated
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())

        # Shuffle and limit
        import random
        random.shuffle(actions)
        return actions[:30]

    def _random_action(self):
        """Generate random action"""
        import random
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }

    def simulate_action(self, balls, table, action):
        """Execute action with noise injection.

        Args:
            balls: Ball state dictionary
            table: Table object
            action: Action dictionary

        Returns:
            Shot object after simulation
        """
        import pooltool as pt

        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            # Inject Gaussian noise
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def analyze_shot_for_reward(self, shot, last_state, player_targets):
        """Analyze shot result and calculate reward score.

        Args:
            shot: Shot object after simulation
            last_state: Ball state before shot
            player_targets: Target ball list for current player

        Returns:
            float: Reward score
        """
        # Basic analysis
        new_pocketed = [bid for bid, b in shot.balls.items()
                        if b.state.s == 4 and last_state[bid].state.s != 4]

        # Determine ball ownership
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed
                         if bid not in player_targets and bid not in ["cue", "8"]]

        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        # Analyze first contact
        first_contact_ball_id = None
        foul_first_hit = False
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        # Foul判定
        if first_contact_ball_id is None:
            if len(last_state) > 2 or player_targets != ['8']:
                foul_first_hit = True
        else:
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True

        # Analyze cushion hits
        cue_hit_cushion = False
        target_hit_cushion = False
        foul_no_rail = False

        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True

        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            foul_no_rail = True

        # Calculate reward
        score = 0

        if cue_pocketed and eight_pocketed:
            score -= 500
        elif cue_pocketed:
            score -= 100
        elif eight_pocketed:
            is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
            score += 150 if is_targeting_eight_ball_legally else -500

        # Special rule: hitting wrong ball when only 8-ball left
        only_eight_ball_left = (len(player_targets) == 1 and player_targets[0] == "8")
        hit_wrong_ball_when_only_eight = (only_eight_ball_left and first_contact_ball_id is not None
                                           and first_contact_ball_id != "8")

        if foul_first_hit:
            if hit_wrong_ball_when_only_eight:
                score -= 15
            else:
                score -= 30

        if foul_no_rail:
            score -= 30

        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20

        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            score = 10

        return score

    def _denormalize_action(self, action_norm):
        """Denormalize action from [0,1] to physical range"""
        action_norm = np.clip(action_norm, 0.0, 1.0)
        return action_norm * (self.action_max - self.action_min) + self.action_min

    def _balls_state_to_81(self, balls_state, my_targets=None, table=None):
        """Convert ball state to 81-dimensional vector.

        Args:
            balls_state: Ball state dictionary
            my_targets: Target ball list
            table: Table object

        Returns:
            np.ndarray: 81-dimensional state vector
        """
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

            if ball_id in balls_state:
                ball = balls_state[ball_id]
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

        # Target ball one-hot
        if my_targets:
            for t in my_targets:
                if t.isdigit():
                    idx = int(t) - 1
                    if 0 <= idx <= 14:
                        state[66 + idx] = 1.0

        return state

    def _state_seq_to_tensor(self, state_seq):
        """Convert state sequence to tensor"""
        if len(state_seq) != 3:
            raise ValueError(f"state_seq length must be 3, got {len(state_seq)}")

        for i, s in enumerate(state_seq):
            if not isinstance(s, np.ndarray) or s.shape != (81,):
                raise TypeError(
                    f"state_seq[{i}] must be np.ndarray(81), got {type(s)} {getattr(s, 'shape', None)}"
                )

        states = np.stack(state_seq, axis=0)
        states = self.state_preprocessor(states)
        return torch.from_numpy(states).float()

    def _expand_and_evaluate(self, node, balls, table, player_targets, root_player, depth, remaining_hits):
        """Expand node and evaluate.

        Returns:
            float: Evaluated value
        """
        # Check losing conditions
        cue_ball = balls.get('cue')
        cue_in_pocket = cue_ball and cue_ball.state.s == 4

        eight_ball = balls.get('8')
        eight_in_pocket = eight_ball and eight_ball.state.s == 4

        my_targets = player_targets[root_player]
        non_eight_targets = [bid for bid in my_targets if bid != '8']
        has_non_eight_targets_left = any(bid in balls and balls[bid].state.s != 4
                                         for bid in non_eight_targets)

        # Losing conditions
        if (cue_in_pocket and eight_in_pocket) or (eight_in_pocket and has_non_eight_targets_left):
            return 0.0

        # Depth limit check
        if depth >= remaining_hits:
            state_tensor = self._state_seq_to_tensor(node.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(state_tensor)
                value = out["value_output"].item()

            return value

        # Generate candidate actions
        candidate_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table)
        n_candidates = len(candidate_actions)

        # Check if only 8-ball remains
        remaining_targets = [bid for bid in player_targets[root_player] if balls[bid].state.s != 4]
        has_only_eight_ball = len(remaining_targets) == 1 and remaining_targets[0] == '8'

        # Get model policy
        state_tensor = self._state_seq_to_tensor(node.state_seq)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(state_tensor)
            policy_output = out["policy_output"][0].cpu().numpy()
            value_output = out["value_output"].item()

        # Denormalize model action
        model_action = self._denormalize_action(policy_output)

        # Action filtering based on distance to model output
        action_distances = []
        for action in candidate_actions:
            action_arr = np.array([action['V0'], action['phi'], action['theta'], action['a'], action['b']])

            phi_diff = abs(action_arr[1] - model_action[1])
            if phi_diff > 180:
                phi_diff = 360 - phi_diff

            v0_diff = abs(action_arr[0] - model_action[0])
            distance = (phi_diff / 180.0) * 0.7 + (v0_diff / (8.0 - 0.5)) * 0.3

            action_distances.append((action, distance))

        action_distances.sort(key=lambda x: x[1])

        keep_count = max(1, int(self.n_simulations * 2 / 3))
        filtered_actions = [action for action, distance in action_distances[:keep_count]]

        # Add 8-ball shots if needed
        if has_only_eight_ball:
            eight_ball_actions = self.generate_heuristic_actions(
                balls, player_targets[root_player], table, only_eight_ball=True)
            for eight_action in eight_ball_actions:
                if eight_action not in filtered_actions:
                    filtered_actions.append(eight_action)

        # Evaluate each action
        best_value = -float('inf')

        for action in filtered_actions:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = self.simulate_action(balls, table, action)

            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = self.analyze_shot_for_reward(shot, last_state_snapshot, player_targets[root_player])

            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            depth_factor = depth / remaining_hits if remaining_hits > 0 else 1.0
            value = depth_factor * value_output + (1 - depth_factor) * normalized_reward

            # Recursive expansion
            if shot is not None and raw_reward > -500 and normalized_reward > 0 and (depth + 1) < remaining_hits:
                new_balls_state = {bid: ball for bid, ball in shot.balls.items()}
                new_state_vec = self._balls_state_to_81(new_balls_state, my_targets=player_targets[root_player], table=table)
                new_state_seq = node.state_seq[1:] + [new_state_vec]
                child_node = MCTSNode(new_state_seq, parent=node)
                child_value = self._expand_and_evaluate(child_node, shot.balls, table, player_targets, root_player, depth + 1, remaining_hits)
                value += child_value * 0.9

            if value > best_value:
                best_value = value

        return best_value

    def search(self, state_seq, balls, table, player_targets, root_player, remaining_hits):
        """Perform MCTS search.

        Args:
            state_seq: State sequence (list of 3 x 81-dim numpy arrays)
            balls: Ball state dictionary
            table: Table object
            player_targets: Player target ball dictionary {player: [ball_ids]}
            root_player: Current player ('A' or 'B')
            remaining_hits: Remaining hits in game

        Returns:
            np.ndarray: Best action [V0, phi, theta, a, b]
        """
        root = MCTSNode(state_seq)

        # Generate candidate actions
        candidate_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table)
        n_candidates = len(candidate_actions)

        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)

        # Calculate max depth
        current_max_depth = min(self.max_depth, remaining_hits)

        # Search start time
        start_time = time.time()

        # MCTS loop
        simulation_count = 0
        time_exceeded = False

        for _ in range(self.n_simulations):
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_search_time:
                time_exceeded = True
                print(f"[Multi-step MCTS] Search timeout ({elapsed_time:.2f}s > {self.max_search_time}s)")
                break

            simulation_count += 1

            # Selection (UCB)
            if np.sum(N) < n_candidates:
                idx = int(np.sum(N))
            else:
                ucb_values = Q + self.c_puct * np.sqrt(np.log(np.sum(N) + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)

            action = candidate_actions[idx]

            # Model evaluation
            state_tensor = self._state_seq_to_tensor(root.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(state_tensor)
                value_output = out["value_output"].item()

            # Simulation with noise
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = self.simulate_action(balls, table, action)

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = self.analyze_shot_for_reward(shot, last_state_snapshot, player_targets[root_player])

            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            depth = 0
            depth_factor = depth / current_max_depth if current_max_depth > 0 else 1.0
            value = depth_factor * value_output + (1 - depth_factor) * normalized_reward

            # Backpropagation
            N[idx] += 1
            Q[idx] += (value - Q[idx]) / N[idx]

        # Handle unsearched actions on timeout
        if time_exceeded:
            state_tensor = self._state_seq_to_tensor(root.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(state_tensor)
                model_value = out["value_output"].item()

            for idx in range(n_candidates):
                if N[idx] == 0:
                    Q[idx] = model_value
                    N[idx] = 1

        # Final decision
        avg_rewards = Q
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]

        print(f"[Multi-step MCTS] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {simulation_count}/{self.n_simulations})")

        return np.array([best_action['V0'], best_action['phi'], best_action['theta'], best_action['a'], best_action['b']], dtype=np.float32)
