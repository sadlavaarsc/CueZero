import tensorflow as tf

def policy_value_loss(targets, policy_pred, value_pred):
    """Policy-value loss function"""
    # Policy loss (mean squared error)
    policy_loss = tf.reduce_mean(tf.square(targets['policy'] - policy_pred))
    
    # Value loss (mean squared error)
    value_loss = tf.reduce_mean(tf.square(targets['value'] - value_pred))
    
    # Total loss
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss