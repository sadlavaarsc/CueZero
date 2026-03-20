import numpy as np

class ReplayBuffer:
    """Replay buffer for experience replay"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch from buffer"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in batch]
    
    def __len__(self):
        """Get buffer size"""
        return len(self.buffer)