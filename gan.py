import tensorflow as tf
import numpy as np
import pickle
from utils import *


class GAN:

    def __init__(self, dataset, batch_size=32, noise_dim=64, epochs=5, learning_rate=1e-4):
        self._dataset = dataset
        self._batch_size = batch_size
        self._noise_dim = noise_dim
        self._learning_rate = learning_rate
        self._epochs = epochs        
        self._build_model()

    def train(self, session):
        session.run(tf.global_variables_initializer())
        test_noise = np.random.uniform(0, 1, [self._batch_size, self._noise_dim])

        for epoch in range(self._epochs):
            num_batches = self._dataset.num_trains // self._batch_size

            for i in range(num_batches):    
                batch = self._dataset._next_batch(self._batch_size) 
                noise = np.random.uniform(0, 1, [self._batch_size, self._noise_dim])                                           

                # Train Discriminator
                feed_dict = { self._real_input: batch, self._noise: noise, self._is_train: True }
                _, d_error = session.run([self._d_opt, self._d_loss], feed_dict=feed_dict)

                # Train Generator
                feed_dict = { self._noise: noise, self._is_train: True }
                _, g_error = session.run([self._g_opt, self._g_loss], feed_dict=feed_dict)
        
            print("Epoch: {0}, d_loss: {1}, g_loss: {2}".format(epoch, d_error, g_error))
            
        feed_dict = {self._noise: test_noise, self._is_train: False}
        samples = session.run(self._g_sample, feed_dict=feed_dict)
        show_images(samples[:16])

    def _build_model(self):
        self._init_variables()

        self._g_sample = self._generator(self._noise)
        d_logit_real = self._discriminator(self._real_input)
        d_logit_fake = self._discriminator(self._g_sample, reuse=True)

        # Losses
        d_loss_real = self._loss(logits=d_logit_real, labels=tf.ones_like(d_logit_real))
        d_loss_fake = self._loss(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake))
        self._d_loss = tf.add(d_loss_real, d_loss_fake)
        self._g_loss = self._loss(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake))

        # Optimizers
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')
        self._d_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.5).minimize(self._d_loss, var_list=d_vars)
        self._g_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.5).minimize(self._g_loss, var_list=g_vars)

    def _init_variables(self): 
        self._real_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self._noise = tf.placeholder(tf.float32, [None, self._noise_dim])
        self._is_train = tf.placeholder(tf.bool)

    def _discriminator(self, input, name='d_', reuse=None):
        # We have multiple instances of the discriminator in the same computation graph,
        # so set variable sharing if this is not the first invocation of this function.
        with tf.variable_scope(name, reuse = reuse):
            dis_conv1 = conv2d(input, 4, 2, 32, 'conv1')
            dis_lrelu1 = leaky_relu(dis_conv1)
            dis_conv2 = conv2d(dis_lrelu1, 4, 2, 64, 'conv2')
            dis_batchnorm2 = batch_norm(dis_conv2, self._is_train)
            dis_lrelu2 = leaky_relu(dis_batchnorm2)
            dis_conv3 = conv2d(dis_lrelu2, 4, 2, 128, 'conv3')
            dis_batchnorm3 = batch_norm(dis_conv3, self._is_train)
            dis_lrelu3 = leaky_relu(dis_batchnorm3)
            dis_reshape3 = tf.reshape(dis_lrelu3, [-1, 4 * 4 * 128])
            dis_fc4 = fc(dis_reshape3, 1, 'fc4')
            return dis_fc4

    def _generator(self, input, name='g_'):
        with tf.variable_scope(name):
            gen_fc1 = fc(input, 4 * 4 * 128, 'fc1')
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 4, 4, 128])
            gen_batchnorm1 = batch_norm(gen_reshape1, self._is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)
            gen_conv2 = conv2d_transpose(gen_lrelu1, 4, 2, 64, 'conv2')
            gen_batchnorm2 = batch_norm(gen_conv2, self._is_train)
            gen_lrelu2 = leaky_relu(gen_batchnorm2)
            gen_conv3 = conv2d_transpose(gen_lrelu2, 4, 2, 32, 'conv3')
            gen_batchnorm3 = batch_norm(gen_conv3, self._is_train)
            gen_lrelu3 = leaky_relu(gen_batchnorm3)
            gen_conv4 = conv2d_transpose(gen_lrelu3, 4, 2, 1, 'conv4')
            gen_sigmoid4 = tf.sigmoid(gen_conv4)
            return gen_sigmoid4

    def _loss(self, logits, labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)

    def _sample_noise(self, shape=(1, 1)):
        return 

