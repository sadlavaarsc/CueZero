#!/usr/bin/env python3
"""Self-play script for generating training data"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuezero.env.billiards_env import BilliardsEnv
from cuezero.mcts.search import MCTS
from cuezero.models.policy_network import PolicyNetwork
from cuezero.models.value_network import ValueNetwork
from cuezero.selfplay.selfplay_worker import SelfPlayWorker
from cuezero.selfplay.dataset_builder import DatasetBuilder
from cuezero.utils.config import load_config
from cuezero.utils.logger import setup_logger

def main():
    # Setup logger
    logger = setup_logger('selfplay')

    # Load configuration
    config = load_config('configs/mcts.yaml')

    # Initialize environment
    env = BilliardsEnv()

    # Initialize networks
    input_dim = 22  # Example input dimension
    policy_output_dim = 3  # Example output dimension
    policy_network = PolicyNetwork(input_dim, policy_output_dim)
    value_network = ValueNetwork(input_dim)

    # Initialize MCTS
    mcts = MCTS(policy_network, value_network, env)

    # Initialize self-play worker
    selfplay_worker = SelfPlayWorker(env, mcts, policy_network, value_network)

    # Initialize dataset builder
    dataset_builder = DatasetBuilder(buffer_size=10000)

    # Generate self-play data
    num_games = 100
    logger.info(f'Generating {num_games} self-play games...')

    for i in range(num_games):
        game_data = selfplay_worker.play_game()
        dataset_builder.add_game_data(game_data)
        logger.info(f'Game {i+1}/{num_games} completed')

    # Save dataset
    logger.info(f'Dataset size: {dataset_builder.get_buffer_size()}')
    
    # Save dataset to disk
    import json
    import os
    from datetime import datetime
    
    # Create output directory
    output_dir = 'data/selfplay'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'selfplay_data_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Save data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset_builder.buffer, f, indent=2, ensure_ascii=False)

    logger.info('Self-play completed successfully')

if __name__ == '__main__':
    main()