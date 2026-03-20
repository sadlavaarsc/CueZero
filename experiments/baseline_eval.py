#!/usr/bin/env python3
"""Baseline evaluation script"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuezero.env.billiards_env import BilliardsEnv
from cuezero.inference.agent import Agent
from cuezero.utils.logger import setup_logger

# Setup logger
logger = setup_logger('baseline_eval')

class BasicAgent(Agent):
    """Basic baseline agent"""
    
    def get_action(self, state):
        """Get action based on simple heuristic"""
        # TODO: Implement basic agent logic
        import numpy as np
        return np.random.uniform(-1, 1, 3)

class BasicAgentPro(Agent):
    """Improved baseline agent"""
    
    def get_action(self, state):
        """Get action based on improved heuristic"""
        # TODO: Implement improved agent logic
        import numpy as np
        return np.random.uniform(-1, 1, 3)

# Evaluation function
def evaluate_baseline_agents():
    """Evaluate baseline agents"""
    env = BilliardsEnv()
    basic_agent = BasicAgent()
    basic_agent_pro = BasicAgentPro()
    
    # TODO: Implement evaluation against trained agents
    logger.info('Evaluating baseline agents...')
    
    # Example evaluation
    num_games = 100
    basic_wins = 0
    pro_wins = 0
    draws = 0
    
    for i in range(num_games):
        state = env.reset()
        done = False
        current_agent = basic_agent if i % 2 == 0 else basic_agent_pro
        
        while not done:
            action = current_agent.get_action(state)
            state, reward, done, info = env.step(action)
            
            # Switch agent
            current_agent = basic_agent_pro if current_agent == basic_agent else basic_agent
        
        # Determine winner
        if reward > 0:
            if i % 2 == 0:
                basic_wins += 1
            else:
                pro_wins += 1
        elif reward < 0:
            if i % 2 == 0:
                pro_wins += 1
            else:
                basic_wins += 1
        else:
            draws += 1
        
        logger.info(f'Game {i+1}/{num_games} completed')
    
    logger.info(f'BasicAgent wins: {basic_wins}')
    logger.info(f'BasicAgentPro wins: {pro_wins}')
    logger.info(f'Draws: {draws}')
    logger.info(f'BasicAgent win rate: {basic_wins / (basic_wins + pro_wins) * 100:.2f}%')

if __name__ == '__main__':
    evaluate_baseline_agents()