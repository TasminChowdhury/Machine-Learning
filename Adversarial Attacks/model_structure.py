"""
Models used to train in MNIST and CIFAR10.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf
from utils_tf import pixel2phase
from cleverhans.model import Model


class HeReLuNormalInitializer(tf.initializers.random_normal):
    def __init__(self, dtype=tf.float32):
        self.dtype = tf.as_dtype(dtype)

    def get_config(self):
        return dict(dtype=self.dtype.name)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info
        dtype = self.dtype if dtype is None else dtype
        std = tf.rsqrt(tf.cast(tf.reduce_prod(shape[:-1]), tf.float32) + 1e-7)
        return tf.random_normal(shape, stddev=std, dtype=dtype)


class ModelDense(Model):
    def __init__(self, scope, nb_classes, reg, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.reg = reg

    def fprop(self, x, **kwargs):
        del kwargs
        my_conv = functools.partial(tf.layers.dense, activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg),
                                    kernel_initializer=HeReLuNormalInitializer,
                                    )
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = pixel2phase(x)
            y = tf.layers.flatten(y)
            y = my_conv(y, 800)
            y = my_conv(y, 300)

            logits = tf.layers.dense(y, self.nb_classes,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg),
                                     kernel_initializer=HeReLuNormalInitializer)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}
