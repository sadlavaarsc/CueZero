import tensorflow as tf
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork

class PolicyValueNetwork(tf.keras.Model):
    """Combined policy-value network for billiards"""
    
    def __init__(self, input_dim, policy_output_dim):
        super(PolicyValueNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu')
        ])
        
        # Policy head
        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(policy_output_dim, activation='tanh')
        ])
        
        # Value head
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
    
    def call(self, inputs):
        """Forward pass"""
        features = self.feature_extractor(inputs)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
    
    def get_config(self):
        """Get model configuration"""
        config = super(PolicyValueNetwork, self).get_config()
        return config