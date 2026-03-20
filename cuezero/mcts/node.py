import numpy as np

class Node:
    """MCTS node for billiards"""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0
    
    def is_leaf(self):
        """Check if node is a leaf"""
        return len(self.children) == 0
    
    def is_root(self):
        """Check if node is root"""
        return self.parent is None
    
    def add_child(self, child_node):
        """Add child node"""
        self.children.append(child_node)
    
    def update(self, value):
        """Update node statistics"""
        self.visits += 1
        self.value += value
    
    def get_uct_score(self, exploration_weight=0.25):
        """Calculate UCT score"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * self.prior * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration