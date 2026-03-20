#!/usr/bin/env python3
"""Training script"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuezero.models.networks import PolicyValueNetwork
from cuezero.training.trainer import Trainer
from cuezero.training.loss import policy_value_loss
from cuezero.selfplay.dataset_builder import DatasetBuilder
from cuezero.utils.config import load_config
from cuezero.utils.logger import setup_logger
import tensorflow as tf

def main():
    # Setup logger
    logger = setup_logger('train')

    # Load configuration
    config = load_config('configs/training.yaml')

    # Initialize model
    input_dim = 22  # Example input dimension
    policy_output_dim = 3  # Example output dimension
    model = PolicyValueNetwork(input_dim, policy_output_dim)

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])

    # Initialize trainer
    trainer = Trainer(model, policy_value_loss, optimizer)

    # Load dataset
    dataset_builder = DatasetBuilder(buffer_size=config['data']['buffer_size'])
    
    # Load actual data from disk
    import os
    import json
    
    # Check for existing selfplay data
    data_dir = 'data/selfplay'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        dataset_builder.add_game_data(data)
                    logger.info(f'Loaded data from {filename}')
                except Exception as e:
                    logger.error(f'Error loading data from {filename}: {e}')
    else:
        logger.warning('No selfplay data directory found')

    # Train model
    dataset = dataset_builder.build_dataset(batch_size=config['training']['batch_size'])
    if dataset:
        trainer.train(dataset, epochs=config['training']['epochs'])
        trainer.save_model('models/cuezero_model')
    else:
        logger.error('No data available for training')
        sys.exit(1)

    logger.info('Training completed successfully')

if __name__ == '__main__':
    main()