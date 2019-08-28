#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler import Cifar10Sampler
from mlgm.model import Vae
from mlgm.logger import Logger

def main():   
    with tf.Session() as sess:
        latent_dim = 16
        model = Vae(
            encoder_layers=[
                layers.Reshape((32, 32, 3)),
                layers.Conv2D(filters=32,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Conv2D(filters=64,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Conv2D(filters=128,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Flatten(),
                layers.Dense(units=(latent_dim + latent_dim))
            ],
            decoder_layers=[
                layers.Dense(units=4 * 4 * 64, activation=tf.nn.relu),
                layers.Reshape(target_shape=(4, 4, 64)),
                layers.Conv2DTranspose(filters=64,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=64,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="SAME", data_format="channels_last"),
                # layers.Activation('sigmoid'),
            ],
            latent_dim=latent_dim,
            sess=sess)
        model.build(layer_in=tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32),
                    layer_out=tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32))
        logger = Logger("vae_cifar10", save_period=5, std_out_period=1)
        sampler = Cifar10Sampler(
            training_digits=list(range(7)), 
            batch_size=128,
            one_hot_labels=False) 
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        num_epoch = 7
        for itr in range(num_epoch):
            end_of_epoch = False
            while not end_of_epoch:
                x_train, y_train, end_of_epoch = sampler.sample()
                model.optimize(x_train, y_train)
            x_test, y_test = sampler.get_test_set()
            loss = model.compute_loss(x_test, y_test)
            logger.new_summary()
            logger.add_value("loss", loss)
            logger.dump_summary(itr)
        logger.save_tf_variables(model.get_variables(), itr, sess)
      
        logger = Logger("maml_vae_cifar10", save_period=2499)
        
        
if __name__ == "__main__":
    main()
