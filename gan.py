import tensorflow as tf
import numpy as np
import pickle
from utils import show_images


class GAN:

    def __init__(self, dataset, batch_size=128, noise_dim=96, epochs=10):
        self._dataset = dataset
        self._batch_size = batch_size
        self._noise_dim = noise_dim
        self._epochs = epochs        
        self._build_model()

    def train(self, session):
        session.run(tf.global_variables_initializer())

        for epoch in range(self._epochs):
            num_batches = self._dataset.num_trains // self._batch_size

            for i in range(num_batches):    
                batch = self._dataset._next_batch(self._batch_size)                            
                feed_dict = { self._X: batch }

                # Train Discriminator
                _, d_error = session.run([self._d_opt, self._d_loss], feed_dict=feed_dict)

                # Train Generator
                _, g_error = session.run([self._g_opt, self._g_loss])
        
            print("Epoch: {0}, d_loss: {1}, g_loss: {2}".format(epoch, d_error, g_error))
            
        samples = session.run(self._g_sample)
        show_images(samples[:16])

    def _build_model(self):
        # Load data
        #x_train = input_data.read_data_sets('./mnist', one_hot=False)
        #num_train = x_train.train.num_examples
        self._init_variables()

        z = self._sample_noise(shape=(self._batch_size, self._noise_dim))
        self._g_sample = self._generator(z)
        d_logit_real = self._discriminator(self._X)
        d_logit_fake = self._discriminator(self._g_sample, reuse=True)

        # Losses
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self._d_loss = tf.add(d_loss_real, d_loss_fake)
        self._g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

        # Optimizers
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')
        self._d_opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self._d_loss, var_list=d_vars)
        self._g_opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self._g_loss, var_list=g_vars)

    def _init_variables(self): 
        self._X = tf.placeholder(tf.float32, shape=(None, self._dataset.image_size))


    def _generator(self, z, name="g_"):
        with tf.variable_scope(name):
            init = tf.contrib.layers.xavier_initializer()
            l1 = tf.nn.relu(tf.layers.dense(inputs=z,units=1024,kernel_initializer=init,name='1-Layer',use_bias=True))
            l2 = tf.nn.relu(tf.layers.dense(inputs=l1,units=1024,kernel_initializer=init,name='2-Layer',use_bias=True))
            out = tf.layers.dense(inputs=l2,units=784, activation=tf.nn.tanh,kernel_initializer=init,name='3-Layer',use_bias=True)           
        return out
    
    def _discriminator(self, x, name="d_", reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            init = tf.contrib.layers.xavier_initializer()
            l1 = tf.nn.leaky_relu(tf.layers.dense(inputs=x,units=256,kernel_initializer=init,name='1-Layer',use_bias=True), alpha=0.01) 
            l2 = tf.nn.leaky_relu(tf.layers.dense(inputs=l1,units=256,kernel_initializer=init,name='2-Layer',use_bias=True), alpha=0.01)
            logits = tf.layers.dense(inputs=l2,units=1,kernel_initializer=init,name='3-Layer',use_bias=True)                
        return logits

    def _sample_noise(self, shape=(1, 1)):
        return tf.random_uniform(shape=shape, minval=-1, maxval=1)

