from mlgm.model import Model

import numpy as np
import tensorflow as tf


class Vae(Model):
    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 latent_dim,
                 sess,
                 name="vae"):
        self._encoder = Model(encoder_layers, sess, model_name="encoder")
        self._decoder = Model(decoder_layers, sess, model_name="decoder")
        self._name = name
        self._sess = sess

    def _encode(self, layer_in, use_tensors=None):
        mean, logvar = tf.split(
            self._encoder.build_forward_pass(layer_in, use_tensors),
            num_or_size_splits=2,
            axis=1)
        return mean, logvar

    def _reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape, dtype=tf.dtypes.float64)
        return eps * tf.exp(logvar * .5) + mean

    def _decode(self, z, use_tensors=None):
        logits = self._decoder.build_forward_pass(z, use_tensors)
        return logits

    @property
    def mean_sym(self):
        return self._mean_sym

    @property
    def logvar_sym(self):
        return self._logvar_sym

    @property
    def latent_sym(self):
        return self._latent_sym

    def build_forward_pass(self, layer_in, use_tensors=None):
        self._mean_sym, self._logvar_sym = self._encode(layer_in, use_tensors)
        self._latent_sym = self._reparameterize(self._mean_sym,
                                                self._logvar_sym)
        return self._decode(self._latent_sym, use_tensors)

    def _log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.cast(tf.math.log(2. * np.pi), dtype=tf.float64)
        mean = tf.cast(sample, dtype=tf.float64)
        logvar = tf.cast(logvar, dtype=tf.float64)
        return tf.reduce_sum(
            np.float64(-.5) * (
                (sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def build_loss(self, labels, logits):
        # TODO: move reshaping, it's only useful to add the channel for MNIST
        labels = tf.reshape(labels, labels.get_shape().concatenate(1))
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self._log_normal_pdf(self._latent_sym, 0., 0.)
        logqz_x = self._log_normal_pdf(self._latent_sym, self._mean_sym,
                                       self._logvar_sym)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def build_accuracy(self, labels, logits, name=None):
        # TODO: move reshaping, it's only useful to add the channel for MNIST
        with tf.variable_scope(
                name,
                default_name=self._name + "_accuracy",
                values=[labels, logits]):
            labels = tf.reshape(labels, labels.get_shape().concatenate(1))
            return tf.reduce_mean(tf.abs(labels - logits))
