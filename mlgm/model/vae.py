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
        eps = tf.random.normal(shape=mean.shape)
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
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def build_loss(self, labels, logits):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self._log_normal_pdf(self._latent_sym, 0., 0.)
        logqz_x = self._log_normal_pdf(self._latent_sym, self._mean_sym,
                                       self._logvar_sym)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def build_accuracy(self, labels, logits, name=None):
        return 0.

    def get_variables(self):
        all_vars = []
        all_vars.extend(self._encoder.get_variables())
        all_vars.extend(self._decoder.get_variables())
        return all_vars

    def build_gradients(self, loss_sym, fast_params=None):
        grads = {}
        params = {}
        if not fast_params:
            enc_grads, enc_params = self._encoder.build_gradients(
                loss_sym, fast_params)
            dec_grads, dec_params = self._decoder.build_gradients(
                loss_sym, fast_params)
            grads.update(enc_grads)
            grads.update(dec_grads)
            params.update(enc_params)
            params.update(dec_params)
        else:
            grads, params = super(Vae, self).build_gradients(
                loss_sym, fast_params)
        return grads, params

    def restore_model(self, save_path):
        var_list = [
            var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ]
        saver = tf.train.Saver(var_list)
        saver.restore(self._sess, save_path)
