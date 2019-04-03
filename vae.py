import tensorflow as tf
from mlp import mlp


class VAE:
    def __init__(self,
                 input_dim,
                 name="VAE",
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=1e-4,
                 encoder_hidden_sizes=(256, 128, 64),
                 decoder_hidden_sizes=(64, 128, 256),
                 hidden_nonlinearity=tf.nn.elu,
                 latent_dim=16):
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.latent_dim = latent_dim
        self.name = name
        self._variable_scope = None

        self.losses, self.train_op = self.build()

    def build(self):
        with tf.variable_scope(self.name, reuse=False) as self._variable_scope:
            self.x = tf.placeholder(tf.float32, shape=(None, self.input_dim))

            self.encoder = mlp(
                input_var=self.x,
                output_dim=self.encoder_hidden_sizes[-1],
                hidden_sizes=self.encoder_hidden_sizes[:-1],
                hidden_nonlinearity=self.hidden_nonlinearity,
                output_nonlinearity=self.hidden_nonlinearity,
                hidden_w_init=tf.zeros_initializer(),
                output_w_init=tf.zeros_initializer(),
                name="encoder")

            self.mu = tf.layers.dense(inputs=self.encoder, units=self.latent_dim,
                    activation=None, name="mu")

            self.log_sigma = tf.layers.dense(inputs=self.encoder, units=self.latent_dim,
                    activation=None, name="log_sigma")

            self.eps = tf.random_normal(
                shape=tf.shape(self.log_sigma),
                mean=0, stddev=1, dtype=tf.float32)
            
            self.z = self.mu + tf.exp(self.log_sigma) * self.eps

            self.x_hat = mlp(
                input_var=self.z,
                output_dim=self.input_dim,
                hidden_sizes=self.decoder_hidden_sizes,
                hidden_nonlinearity=self.hidden_nonlinearity,
                output_nonlinearity=tf.sigmoid,
                hidden_w_init=tf.zeros_initializer(),
                output_w_init=tf.zeros_initializer(),
                name="decoder")

        with tf.name_scope('loss'):
            epsilon = 1e-10
            recon_loss = -tf.reduce_sum(
                self.x * tf.log(epsilon + self.x_hat) + 
                (1 - self.x) * tf.log(epsilon + 1 - self.x_hat), 
                axis=1
            )
            recon_loss = tf.reduce_mean(recon_loss)

            latent_loss = 0.5 * tf.reduce_sum(
                tf.exp(self.log_sigma) + tf.square(self.mu) - 1. - self.log_sigma,
                axis=1)

            latent_loss = tf.reduce_mean(latent_loss)
            total_loss = recon_loss + latent_loss

            losses = {
                'recon_loss': recon_loss,
                'latent_loss': latent_loss,
                'total_loss': total_loss
            }

        with tf.name_scope("op"):
            train_op = self.optimizer(self.learning_rate).minimize(total_loss)

        return losses, train_op

    def optimize(self, batch):
        losses, _ = tf.get_default_session().run([self.losses, self.train_op], feed_dict={self.x: batch})
        return losses

    def reconstruct(self, batch):
       return tf.get_default_session().run(self.x_hat, feed_dict={self.x: batch})

    def reconstruct_from_z(self, z):
       return tf.get_default_session().run(self.x_hat, feed_dict={self.z: z})

    def transform(self, batch):
        return tf.get_default_session().run(self.z, feed_dict={self.x: batch})

    # ================== API for MAML ================
    def get_weights(self):
        weights = dict()
        for x in self._variable_scope.trainable_variables():
            weights[x.name.split(':')[0]] = x
        return weights

    def forward(self, _input, weights, reuse=False):
        """
        reuse: To reuse the batch_norm/layer_norm or not.
        """
        w = 'VAE/{}/{}/kernel'
        b = 'VAE/{}/{}/bias'
        v = 'encoder'
        out = _input
        for h in ['hidden_0', 'hidden_1', 'output']:
            out = tf.matmul(out, weights[w.format(v, h)]) + weights[b.format(v, h)]
            out = self.hidden_nonlinearity(out) 

        mu_w = weights['VAE/mu/kernel']
        mu_b = weights['VAE/mu/bias']
        log_sigma_w = weights['VAE/log_sigma/kernel']
        log_sigma_b = weights['VAE/log_sigma/bias']

        mu_out = tf.matmul(out, mu_w) + mu_b
        log_sigma_out = tf.matmul(out, log_sigma_w) + log_sigma_b

        eps = tf.random_normal(
                shape=tf.shape(log_sigma_out),
                mean=0, stddev=1, dtype=tf.float32)

        out = mu_out + tf.exp(log_sigma_out) * eps

        v = 'decoder'
        for h in ['hidden_0', 'hidden_1', 'hidden_2']:
            out = tf.matmul(out, weights[w.format(v, h)]) + weights[b.format(v, h)]
            out = self.hidden_nonlinearity(out) 

        out = tf.matmul(out, weights[w.format(v, 'output')]) + weights[b.format(v, 'output')]
        x_hat = tf.sigmoid(out)

        return x_hat, mu_out, log_sigma_out

    def loss_func(self, x, x_hat, mu, log_sigma):
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            x * tf.log(epsilon + x_hat) + 
            (1 - x) * tf.log(epsilon + 1 - x_hat), 
            axis=1
        )        
        recon_loss = tf.reduce_mean(recon_loss)

        latent_loss = 0.5 * tf.reduce_sum(
            tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma,
            axis=1)

        latent_loss = tf.reduce_mean(latent_loss)
        total_loss = recon_loss + latent_loss

        return total_loss

    # ====================================================================================================