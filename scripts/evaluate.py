#!/usr/bin/env python3
"""Evaluation script"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuezero.env.billiards_env import BilliardsEnv
from cuezero.inference.agent import MCTSAgent, PolicyAgent
from cuezero.mcts.search import MCTS
from cuezero.models.policy_network import PolicyNetwork
from cuezero.models.value_network import ValueNetwork
from cuezero.utils.logger import setup_logger

def evaluate_agents(agent1, agent2, env, logger, num_games=100):
    """Evaluate two agents against each other"""
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for i in range(num_games):
        state = env.reset()
        done = False
        current_agent = agent1 if i % 2 == 0 else agent2
        
        while not done:
            action = current_agent.get_action(state)
            state, reward, done, info = env.step(action)
            
            # Switch agent
            current_agent = agent2 if current_agent == agent1 else agent1
        
        # Determine winner
        if reward > 0:
            if i % 2 == 0:
                agent1_wins += 1
            else:
                agent2_wins += 1
        elif reward < 0:
            if i % 2 == 0:
                agent2_wins += 1
            else:
                agent1_wins += 1
        else:
            draws += 1
        
        logger.info(f'Game {i+1}/{num_games} completed')
    
    return agent1_wins, agent2_wins, draws

def main():
    # Setup logger
    logger = setup_logger('evaluate')

    # Initialize environment
    env = BilliardsEnv()

    # Initialize networks
    input_dim = 22  # Example input dimension
    policy_output_dim = 3  # Example output dimension
    policy_network = PolicyNetwork(input_dim, policy_output_dim)
    value_network = ValueNetwork(input_dim)

    # Initialize MCTS
    mcts = MCTS(policy_network, value_network, env)

    # Initialize agents
    mcts_agent = MCTSAgent(mcts)
    policy_agent = PolicyAgent(policy_network)

    # Run evaluation
    logger.info('Evaluating MCTSAgent vs PolicyAgent...')
    agent1_wins, agent2_wins, draws = evaluate_agents(mcts_agent, policy_agent, env, logger)

    # Print results
    logger.info(f'MCTSAgent wins: {agent1_wins}')
    logger.info(f'PolicyAgent wins: {agent2_wins}')
    logger.info(f'Draws: {draws}')
    logger.info(f'MCTSAgent win rate: {agent1_wins / (agent1_wins + agent2_wins) * 100:.2f}%')

    logger.info('Evaluation completed successfully')

if __name__ == '__main__':
    main()