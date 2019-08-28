"""
This module creates a Dropout layer. 

tf.keras.layers.droput would raise an InvalidArugmentError.
See: https://github.com/keras-team/keras/issues/9288
"""
import tensorflow as tf


class Dropout:
    """
    A Droput layer that uses tf.nn.dropout to compute the dropout.

    :params rate: float, the probabily of the element to ouput 0.
    :params seed: int, the random seed used to create random seeds.
    :params name: str, the name for this operation.
    :params noise_shape: 1-D tensor, the shape for randomly generated 
      keep/drop flags.
    """
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
