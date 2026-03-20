import numpy as np
import tensorflow as tf

class DatasetBuilder:
    """Build dataset from self-play data"""
    
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_game_data(self, game_data):
        """Add game data to buffer"""
        self.buffer.extend(game_data)
        
        # Maintain buffer size
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    def build_dataset(self, batch_size=64):
        """Build tensorflow dataset"""
        if not self.buffer:
            return None
        
        # Convert to numpy arrays
        states = np.array([item['state'] for item in self.buffer])
        actions = np.array([item['action'] for item in self.buffer])
        returns = np.array([item['return'] for item in self.buffer])
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((states, {'policy': actions, 'value': returns}))
        dataset = dataset.shuffle(len(self.buffer)).batch(batch_size)
        
        return dataset
    
    def get_buffer_size(self):
        """Get current buffer size"""
        return len(self.buffer)