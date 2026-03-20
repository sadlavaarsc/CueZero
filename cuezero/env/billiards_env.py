import numpy as np
import pooltool as pt
from .physics_wrapper import PhysicsWrapper

class BilliardsEnv:
    """Billiards environment for reinforcement learning"""
    
    def __init__(self):
        self.physics = PhysicsWrapper()
        self.reset()
    
    def reset(self, target_ball='solid'):
        """Reset the environment to initial state"""
        # Initialize table
        self.table = pt.Table.create(pt.TableType.SNOOKER)
        
        # Initialize balls
        self.balls = {
            'cue': pt.Ball.create('cue', pos=(0.5, 0.5)),
            '1': pt.Ball.create('1', pos=(0.7, 0.5)),
            '2': pt.Ball.create('2', pos=(0.7, 0.53)),
            '3': pt.Ball.create('3', pos=(0.7, 0.47)),
            '4': pt.Ball.create('4', pos=(0.73, 0.515)),
            '5': pt.Ball.create('5', pos=(0.73, 0.485)),
            '6': pt.Ball.create('6', pos=(0.76, 0.5)),
            '7': pt.Ball.create('7', pos=(0.76, 0.53)),
            '8': pt.Ball.create('8', pos=(0.76, 0.47)),
            '9': pt.Ball.create('9', pos=(0.79, 0.515)),
            '10': pt.Ball.create('10', pos=(0.79, 0.485)),
            '11': pt.Ball.create('11', pos=(0.82, 0.5)),
            '12': pt.Ball.create('12', pos=(0.82, 0.53)),
            '13': pt.Ball.create('13', pos=(0.82, 0.47)),
            '14': pt.Ball.create('14', pos=(0.85, 0.515)),
            '15': pt.Ball.create('15', pos=(0.85, 0.485))
        }
        
        # Set target ball type
        self.target_ball = target_ball
        
        # Initialize game state
        self.hit_count = 0
        self.current_player = 'A'
        self.done = False
        self.winner = None
        
        # Return observation
        return self.get_observation(self.current_player)
    
    def step(self, action):
        """Take an action and return new state, reward, done, info"""
        # Simulate action
        state = {
            'balls': self.balls,
            'table': self.table
        }
        
        result = self.physics.simulate(state, action)
        self.balls = result['balls']
        self.table = result['table']
        
        # Increment hit count
        self.hit_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if game is done
        done, info = self.get_done()
        
        # Switch player if no balls were pocketed
        if not self._has_pocketed_ball():
            self.current_player = 'B' if self.current_player == 'A' else 'A'
        
        return self.get_observation(self.current_player), reward, done, info
    
    def render(self):
        """Render the environment"""
        # TODO: Implement rendering
        pass
    
    def get_state(self):
        """Get current state"""
        return {
            'balls': self.balls,
            'table': self.table,
            'current_player': self.current_player,
            'hit_count': self.hit_count
        }
    
    def set_state(self, state):
        """Set current state"""
        self.balls = state['balls']
        self.table = state['table']
        self.current_player = state.get('current_player', 'A')
        self.hit_count = state.get('hit_count', 0)
    
    def get_observation(self, player):
        """Get observation for player"""
        # Return balls, targets, and table as observation
        return self.balls, self._get_my_targets(player), self.table
    
    def get_curr_player(self):
        """Get current player"""
        return self.current_player
    
    def take_shot(self, action):
        """Take a shot and return step info"""
        # Simulate action
        state = {
            'balls': self.balls,
            'table': self.table
        }
        
        result = self.physics.simulate(state, action)
        self.balls = result['balls']
        self.table = result['table']
        
        # Increment hit count
        self.hit_count += 1
        
        # Check if any balls were pocketed
        step_info = {}
        pocketed_balls = self._get_pocketed_balls()
        if pocketed_balls:
            step_info['POCKETED_BALLS'] = pocketed_balls
        
        # Check if enemy ball was pocketed
        enemy_balls = self._get_enemy_balls()
        pocketed_enemy = [b for b in pocketed_balls if b in enemy_balls]
        if pocketed_enemy:
            step_info['ENEMY_INTO_POCKET'] = pocketed_enemy
        
        return step_info
    
    def get_done(self):
        """Check if game is done"""
        # Check if 8 ball is pocketed
        if '8' in self.balls and self.balls['8'].state.s == 4:
            self.done = True
            # Determine winner: current player wins when 8 ball is pocketed
            self.winner = self.current_player
            return True, {'winner': self.winner}
        
        # Check if all balls of one type are pocketed
        solid_balls = [b for b in self.balls if b in ['1', '2', '3', '4', '5', '6', '7']]
        stripe_balls = [b for b in self.balls if b in ['9', '10', '11', '12', '13', '14', '15']]
        
        solid_pocketed = all(self.balls[b].state.s == 4 for b in solid_balls)
        stripe_pocketed = all(self.balls[b].state.s == 4 for b in stripe_balls)
        
        if solid_pocketed or stripe_pocketed:
            self.done = True
            # Determine winner: current player wins when all balls of their type are pocketed
            self.winner = self.current_player
            return True, {'winner': self.winner}
        
        return False, {}
    
    def _calculate_reward(self):
        """Calculate reward for current state"""
        reward = 0.0
        
        # Reward for pocketing own balls
        my_targets = self._get_my_targets(self.current_player)
        for target in my_targets:
            if target in self.balls and self.balls[target].state.s == 4:
                reward += 10.0
        
        # Penalty for pocketing enemy balls
        enemy_targets = self._get_enemy_targets(self.current_player)
        for target in enemy_targets:
            if target in self.balls and self.balls[target].state.s == 4:
                reward -= 5.0
        
        # Reward for pocketing 8 ball
        if '8' in self.balls and self.balls['8'].state.s == 4:
            reward += 50.0
        
        return reward
    
    def _get_my_targets(self, player):
        """Get target balls for current player"""
        if self.target_ball == 'solid':
            return ['1', '2', '3', '4', '5', '6', '7']
        else:
            return ['9', '10', '11', '12', '13', '14', '15']
    
    def _get_enemy_targets(self, player):
        """Get enemy target balls"""
        if self.target_ball == 'solid':
            return ['9', '10', '11', '12', '13', '14', '15']
        else:
            return ['1', '2', '3', '4', '5', '6', '7']
    
    def _get_enemy_balls(self):
        """Get enemy balls"""
        return self._get_enemy_targets(self.current_player)
    
    def _has_pocketed_ball(self):
        """Check if any ball was pocketed"""
        for ball in self.balls.values():
            if ball.state.s == 4:
                return True
        return False
    
    def _get_pocketed_balls(self):
        """Get list of pocketed balls"""
        return [bid for bid, ball in self.balls.items() if ball.state.s == 4]