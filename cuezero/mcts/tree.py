from .node import Node

class Tree:
    """MCTS tree for billiards"""
    
    def __init__(self, root_state):
        self.root = Node(root_state)
    
    def get_root(self):
        """Get root node"""
        return self.root
    
    def set_root(self, root_node):
        """Set root node"""
        self.root = root_node
    
    def traverse(self, node, policy_fn, value_fn):
        """Traverse the tree to find leaf node"""
        current = node
        
        while not current.is_leaf():
            # Select child with highest UCT score
            current = max(current.children, key=lambda c: c.get_uct_score())
        
        return current
    
    def backpropagate(self, node, value):
        """Backpropagate value up the tree"""
        current = node
        
        while current is not None:
            current.update(value)
            current = current.parent