import tensorflow as tf

class ValueNetwork(tf.keras.Model):
    """Value network for billiards"""
    
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
    
    def call(self, inputs):
        """Forward pass"""
        return self.model(inputs)
    
    def get_config(self):
        """Get model configuration"""
        config = super(ValueNetwork, self).get_config()
        return config