"""
Tensorflow utils
"""

import tensorflow as tf
from cleverhans.loss import Loss


def fftshift(inputs):
    """
    Calculates and returns fftshift of the inputs tensor along second and third axes.
    :param inputs: Tensor with shape (None, ., ., .)
    :return: a Tensor which is fftshift of inputs along second and third axes with same dtype and shape
    """
    axes = range(1, len(inputs.shape) - 1)
    for k in axes:
        n = inputs.shape[k]
        p2 = (n + 1) // 2
        my_list = tf.concat((tf.range(p2, n), tf.range(p2)), axis=0)
        inputs = tf.gather(inputs, my_list, axis=k)
    return inputs


def pixel2phase(inputs):
    """
    convert the inputs images to the phase domain along each channel.
    :param inputs: Tensor with shape (None, height, width, channels)
    :return: Tensor with same shape and dtype as inputs
    """
    inputs_dtype = inputs.dtype
    dtype = tf.complex64
    inputs = tf.cast(inputs, dtype=dtype)
    input_f = fftshift(tf.transpose(tf.fft2d(tf.transpose(inputs, perm=[0, 3, 1, 2])), perm=[0, 2, 3, 1]))
    input_f = tf.where(tf.less(tf.abs(input_f), 1e-5), tf.zeros(tf.shape(input_f), dtype=dtype), input_f)
    return tf.cast(tf.angle(input_f), dtype=inputs_dtype)


class LossCrossEntropy(Loss):
    def __init__(self, model, smoothing, attack=None, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param smoothing: float, amount of label smoothing for cross-entropy.
        :param attack: function, given an input x, return an attacked x'.
        """
        if smoothing < 0 or smoothing > 1:
            raise ValueError('Smoothing must be in [0, 1]', smoothing)
        del kwargs
        Loss.__init__(self, model, locals(), attack)
        self.smoothing = smoothing

    def fprop(self, x, y, **kwargs):
        if self.attack is not None:
            x = x, self.attack(x)
        else:
            x = x,
        y -= self.smoothing * (y - 1. / tf.cast(y.shape[-1], tf.float32))
        logits = [self.model.get_logits(x, **kwargs) for x in x]
        loss = sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logit)
            for logit in logits) + tf.losses.get_regularization_loss()
        return loss
