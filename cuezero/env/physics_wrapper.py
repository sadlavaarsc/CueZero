import math
import copy
import numpy as np
import pooltool as pt

class PhysicsWrapper:
    """Wrapper for billiards physics simulation"""
    
    def __init__(self):
        # Initialize physics engine
        self.table = pt.Table.prebuilt('Generic Snooker')
        self.balls = {}
    
    def simulate(self, state, action):
        """Simulate the physics given current state and action"""
        # Copy state
        sim_balls = {k: copy.deepcopy(v) for k, v in state['balls'].items()}
        sim_table = copy.deepcopy(state['table'])
        
        # Create cue and shot
        cue = pt.Cue("cue")
        shot = pt.System(
            table=sim_table,
            balls=sim_balls,
            cue=cue
        )
        
        # Set cue state
        cue.set_state(
            V0=action['V0'],
            phi=action['phi'],
            theta=action.get('theta', 0),
            a=action.get('a', 0),
            b=action.get('b', 0)
        )
        
        # Simulate
        try:
            pt.simulate(shot, inplace=True)
            return {
                'balls': shot.balls,
                'table': shot.table,
                'cue': shot.cue
            }
        except Exception:
            return state
    
    def check_collision(self, state):
        """Check for collisions in current state"""
        collisions = []
        balls = state['balls']
        
        # Check ball-ball collisions
        ball_ids = list(balls.keys())
        for i in range(len(ball_ids)):
            for j in range(i + 1, len(ball_ids)):
                ball1 = balls[ball_ids[i]]
                ball2 = balls[ball_ids[j]]
                
                pos1 = ball1.state.rvw[0][:2]
                pos2 = ball2.state.rvw[0][:2]
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < 2 * 0.028575:  # 2 * ball radius
                    collisions.append((ball_ids[i], ball_ids[j]))
        
        return collisions
    
    def calculate_trajectory(self, state, action):
        """Calculate ball trajectory given action"""
        # Simulate and return trajectory
        result = self.simulate(state, action)
        trajectories = {}
        
        for ball_id, ball in result['balls'].items():
            # Extract position history
            trajectory = []
            for state in ball.history.states:
                trajectory.append(state.rvw[0])
            trajectories[ball_id] = trajectory
        
        return trajectories