import tensorflow as tf

class Trainer:
    """Trainer for billiards neural networks"""
    
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train_step(self, states, targets):
        """Single training step"""
        with tf.GradientTape() as tape:
            policy_pred, value_pred = self.model(states)
            loss = self.loss_fn(targets, policy_pred, value_pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def train_epoch(self, dataset):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for states, targets in dataset:
            loss = self.train_step(states, targets)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, dataset, epochs=100):
        """Train for multiple epochs"""
        for epoch in range(epochs):
            loss = self.train_epoch(dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def save_model(self, path):
        """Save model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load model"""
        self.model = tf.keras.models.load_model(path)