import tensorflow as tf

class BN:
    def __init__(self, is_training=True):
        self.is_training = is_training

    def __call__(self, inputs):
        return tf.contrib.slim.batch_norm(inputs, updates_collections=None, is_training=self.is_training,
            scale=True, epsilon=1e-5, decay=0.9)

leaky_relu_alpha = 0.2
def leaky_relu(x, alpha=leaky_relu_alpha, name='Leaky_ReLU'):
    return tf.maximum(x, alpha*x, name=name)

def reshape(inputs, shape):
    return tf.reshape(inputs, shape)

def conv2dtr(inputs, filters, strides=2):
    return tf.contrib.slim.conv2d_transpose(inputs, filters, 5, strides=strides, padding='SAME',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.constant_initializer(0.0))

def conv2d(inputs, filters, strides=2):
    return tf.contrib.slim.conv2d(inputs, filters, 5, strides=strides, padding='SAME',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.constant_initializer(0.0))