# DO NOT USE Keras Dropout as a layer:
# https://github.com/keras-team/keras/issues/9288
import tensorflow as tf


class Dropout:
    def __init__(self, rate, noise_shape=None, seed=None, name=None):
        self._rate = rate
        self._noise_shape = noise_shape
        self._seed = seed
        self._name = name

    def __call__(self, inputs):
        return tf.nn.dropout(
            inputs,
            noise_shape=self._noise_shape,
            seed=self._seed,
            name=self._name,
            rate=self._rate)
