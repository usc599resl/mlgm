import tensorflow as tf
from tensorflow.keras import layers

from mlgm.model import Model
from mlgm.utils import conv_out_size_same


class DcGan(Model):
    def __init__(self,
                 sess,
                 output_height=64,
                 output_width=64,
                 df_dim=64,
                 gf_dim=64,
                 c_dim=3,
                 z_dim=100,
                 name="dcgan"):
        """
        gf_dim: dimension of gen filters in first conv layer
        """
        self._sess = sess
        self._output_height = output_height
        self._output_width = output_width
        self._c_dim = c_dim
        self._z_dim = z_dim
        self._df_dim = df_dim
        self._gf_dim = gf_dim

        self._build_generator()
        self._build_discriminator()

    def _build_discriminator(self):
        discriminator_layers = [
            layers.Conv2D(
                filters=self._df_dim,
                kernel_size=5,
                strides=(2, 2),
                padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(
                filters=self._df_dim * 2,
                kernel_size=5,
                strides=(2, 2),
                padding='same'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(
                filters=self._df_dim * 4,
                kernel_size=5,
                strides=(2, 2),
                padding='same'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(
                filters=self._df_dim * 8,
                kernel_size=5,
                strides=(2, 2),
                padding='same'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((-1, )),
            layers.Dense(units=1),
        ]
        self._discriminator = Model(
            discriminator_layers, self._sess, model_name="discriminator")

    def _build_generator(self):
        s_h, s_w = self._output_height, self._output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        generator_layers = [
            layers.Dense(
                units=self._gf_dim * 8 * s_h16 * s_w16, activation=None),
            layers.Reshape((s_h16, s_w16, self._gf_dim * 8)),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.ReLU(),
            layers.Conv2DTranspose(
                filters=self._gf_dim * 4,
                kernel_size=(s_h8, s_w8),
                strides=(2, 2),
                padding="SAME"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.ReLU(),
            layers.Conv2DTranspose(
                filters=self._gf_dim * 2,
                kernel_size=(s_h4, s_w4),
                strides=(2, 2),
                padding="SAME"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.ReLU(),
            layers.Conv2DTranspose(
                filters=self._gf_dim * 1,
                kernel_size=(s_h2, s_w2),
                strides=(2, 2),
                padding="SAME"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            layers.ReLU(),
            layers.Conv2DTranspose(
                filters=self._c_dim,
                kernel_size=(s_h, s_w),
                strides=(2, 2),
                padding="SAME",
                activation=tf.tanh)
        ]
        self._generator = Model(
            generator_layers, self._sess, model_name="generator")

    def build_forward_pass(self, layer_in, use_tensors=None, training=False):
        layer_in = tf.cast(layer_in, dtype=tf.float32)
        z = tf.random.uniform((
            layer_in.get_shape().as_list()[0],
            self._z_dim,
        ),
                              minval=-1,
                              maxval=1)
        gen_out = self._generator.build_forward_pass(z, use_tensors, training)
        dis_fake_logits = self._discriminator.build_forward_pass(
            gen_out, use_tensors, training)
        dis_real_logits = self._discriminator.build_forward_pass(
            layer_in, use_tensors, training)
        return gen_out, dis_fake_logits, dis_real_logits

    def build_loss(self, labels, out):
        dis_fake_logits = out[1]
        dis_real_logits = out[2]
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dis_real_logits, labels=tf.ones_like(dis_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dis_fake_logits, labels=tf.zeros_like(dis_fake_logits)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dis_fake_logits, labels=tf.ones_like(dis_fake_logits)))
        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss

    def build_gradients(self, loss_sym, fast_params=None):
        d_loss = loss_sym[0]
        g_loss = loss_sym[1]
        grads = {}
        params = {}

        if not fast_params:
            gen_grads, gen_params = self._generator.build_gradients(
                g_loss, fast_params)
            dis_grads, dis_params = self._discriminator.build_gradients(
                d_loss, fast_params)
            grads.update(gen_grads)
            grads.update(dis_grads)
            params.update(gen_params)
            params.update(dis_params)
        else:
            grads, params = super(DcGan, self).build_gradients(
                loss_sym, fast_params)
        return grads, params

    def get_variables(self):
        variables = {}
        variables["generator"] = self._generator.get_variables()
        variables["discriminator"] = self._discriminator.get_variables()
        return variables
