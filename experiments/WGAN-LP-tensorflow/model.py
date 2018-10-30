import tensorflow as tf
from typing import Callable
from operations import *

slim = tf.contrib.slim

class Model(object):
    def __init__(self, input_tensor: tf.Variable,
                 variable_scope_name: str,
                 reuse: bool,
                 img_shape=(64, 64, 4),
                 zsize=128,
                 is_training=True): # TODO : Change from True

        self.input = input_tensor
        self.is_training = is_training
        self.variable_scope_name = variable_scope_name
        self.img_shape = img_shape
        self.zsize = zsize
        self.output_tensor = None
        self.var_list = None
        self.reuse = reuse


class Generator(Model):
    
    def __init__(self, input_tensor: tf.Variable, variable_scope_name: str='Generator', reuse: bool=False):        
        super(Generator, self).__init__(input_tensor, variable_scope_name, reuse)
        self.define_model()

    def define_model(self):
        with tf.variable_scope(self.variable_scope_name, reuse=self.reuse) as vs:
            x, bn = self.input, BN(self.is_training)
            minirows = self.img_shape[0] // 32
            minicols = self.img_shape[1] // 32 
            
            x = slim.fully_connected(inputs=x, num_outputs=minirows * minicols * 512, activation_fn=None)
            x = __leaky_relu__(bn(reshape(x, (tf.shape(x)[0], minirows, minicols, 512))))
            x = __leaky_relu__(bn(conv2dtr(x, 512)))
            x = __leaky_relu__(bn(conv2dtr(x, 256)))
            x = __leaky_relu__(bn(conv2dtr(x, 128)))
            x = __leaky_relu__(bn(conv2dtr(x, 65)))

            self.output_tensor = tf.nn.tanh(conv2dtr(x, self.img_shape[2]))
            self.var_list = tf.contrib.framework.get_variables(vs)


class Critic(Model):
    def __init__(self,
                 input_tensor: tf.Variable,
                 variable_scope_name: str='Critic',
                 reuse: bool=False):

        super(Critic, self).__init__(input_tensor, variable_scope_name, reuse)
        self.define_model()


    def define_model(self):
        with tf.variable_scope(self.variable_scope_name, reuse=self.reuse) as vs:
            x, bn = self.input, BN(True)
            x = __leaky_relu__(conv2d(x, 64))
            x = __leaky_relu__(bn(conv2d(x, 128)))
            x = __leaky_relu__(bn(conv2d(x, 256)))
            x = __leaky_relu__(bn(conv2d(x, 512)))
            x = __leaky_relu__(bn(conv2d(x, 1024)))

            x = slim.flatten(x)
            logits = slim.fully_connected(inputs=x, num_outputs=1, activation_fn=None)
            classification = sigmoid(logits)

            self.output_tensor = logits
            self.var_list = tf.contrib.framework.get_variables(vs)
