import tensorflow as tf
import numpy as np
import pickle
from utils import *


class GAN:

    def __init__(self, dataset, batch_size=64, noise_dim=64, epochs=10, learning_rate=2e-4):
        self._dataset = dataset
        self._batch_size = batch_size
        self._noise_dim = noise_dim
        self._learning_rate = learning_rate
        self._epochs = epochs        
        self._build_model()

    def train(self, session):
        session.run(tf.global_variables_initializer())
        test_noise = np.random.uniform(0., 1., [self._batch_size, self._noise_dim])

        for epoch in range(self._epochs):
            num_batches = self._dataset.num_trains // self._batch_size

            for _ in range(num_batches):             
                train_d = True
                train_g = True

                batch = self._dataset._next_batch(self._batch_size) 
                noise = np.random.uniform(0, 1, [self._batch_size, self._noise_dim])

                feed_dict = { self._real_input: batch, self._noise: noise, self._is_train: True, self._keep_prob: 0.6 }
                d_loss, g_loss = session.run([self._d_loss, self._g_loss], feed_dict=feed_dict)
                
                if g_loss * 1.5 < d_loss:
                    train_g = False                
                if d_loss * 2 < g_loss:
                    train_d = False                                           

                # Train Discriminator
                if train_d:
                    feed_dict = { self._real_input: batch, self._noise: noise, self._is_train: True, self._keep_prob: 0.6 }
                    session.run([self._d_opt], feed_dict=feed_dict)

                # Train Generator
                if train_g:
                    feed_dict = { self._noise: noise, self._is_train: True, self._keep_prob: 0.6 }
                    session.run([self._g_opt], feed_dict=feed_dict)                
                        
            print("Epoch: {0}, d_loss: {1}, g_loss: {2}".format(epoch, d_loss, g_loss))                
            feed_dict = {self._noise: test_noise, self._is_train: False, self._keep_prob: 1.0}
            samples = session.run(self._g_sample, feed_dict=feed_dict)
            show_images(samples)

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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._d_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.5).minimize(self._d_loss, var_list=d_vars)
            self._g_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.5).minimize(self._g_loss, var_list=g_vars)             


    def _init_variables(self): 
        image_size = self._dataset.image_size
        self._real_input = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
        self._noise = tf.placeholder(tf.float32, [None, self._noise_dim])
        self._is_train = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(dtype=tf.float32)

    def _discriminator(self, input, name='d_', reuse=None):
        with tf.variable_scope(name, reuse = reuse):            
            conv1 = tf.layers.conv2d(input, kernel_size=5, filters=64, strides=2, padding='same', activation=leaky_relu)
            drop1 = tf.layers.dropout(conv1, self._keep_prob)
            conv2 = tf.layers.conv2d(drop1, kernel_size=5, filters=64, strides=1, padding='same', activation=leaky_relu)
            drop2 = tf.layers.dropout(conv2, self._keep_prob)
            conv3 = tf.layers.conv2d(drop2, kernel_size=5, filters=64, strides=1, padding='same', activation=leaky_relu)
            drop3 = tf.layers.dropout(conv3, self._keep_prob)
            flatten = tf.contrib.layers.flatten(drop3)
            fc1 = tf.layers.dense(flatten, units=128, activation=leaky_relu)
            out = tf.layers.dense(fc1, units=1)

            return out

    def _generator(self, input, name='g_'):
        with tf.variable_scope(name):
            d1 = 4
            d2 = 1
            activation = leaky_relu
            momentum = 0.99
            x = tf.layers.dense(input, units=d1 * d1 * d2, activation=activation)
            x = tf.layers.dropout(x, self._keep_prob)      
            x = tf.contrib.layers.batch_norm(x, is_training=self._is_train, decay=momentum)  
            x = tf.reshape(x, shape=[-1, d1, d1, d2])
            x = tf.image.resize_images(x, size=[7, 7])
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
            x = tf.layers.dropout(x, self._keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self._is_train, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
            x = tf.layers.dropout(x, self._keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self._is_train, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
            x = tf.layers.dropout(x, self._keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self._is_train, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
            return x

    def _loss(self, logits, labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)
