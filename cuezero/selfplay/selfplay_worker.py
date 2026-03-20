import numpy as np

class SelfPlayWorker:
    """Self-play worker for generating training data"""
    
    def __init__(self, env, mcts, policy_network, value_network):
        self.env = env
        self.mcts = mcts
        self.policy_network = policy_network
        self.value_network = value_network
    
    def play_game(self):
        """Play a self-play game"""
        state = self.env.reset()
        game_data = []
        done = False
        
        while not done:
            # Perform MCTS search
            action = self.mcts.search(state)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            game_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            state = next_state
        
        # Process game data
        processed_data = self.process_game_data(game_data)
        return processed_data
    
    def process_game_data(self, game_data):
        """Process game data to create training examples"""
        processed_data = []
        
        # Calculate returns
        returns = self.calculate_returns(game_data)
        
        for i, data in enumerate(game_data):
            processed_data.append({
                'state': data['state'],
                'action': data['action'],
                'return': returns[i]
            })
        
        return processed_data
    
    def calculate_returns(self, game_data, discount_factor=0.99):
        """Calculate returns for each step"""
        returns = []
        R = 0
        
        # Iterate from the end of the game
        for data in reversed(game_data):
            R = data['reward'] + discount_factor * R
            returns.insert(0, R)
        
        return returns