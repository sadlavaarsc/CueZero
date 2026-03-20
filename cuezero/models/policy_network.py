import tensorflow as tf

class PolicyNetwork(tf.keras.Model):
    """Policy network for billiards"""
    
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='tanh')
        ])
    
    def call(self, inputs):
        """Forward pass"""
        return self.model(inputs)
    
    def get_config(self):
        """Get model configuration"""
        config = super(PolicyNetwork, self).get_config()
        return config