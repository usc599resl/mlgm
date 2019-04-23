#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler import Cifar10MetaSampler
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Vae
from mlgm.logger import Logger

def main():
    metasampler = Cifar10MetaSampler(
        batch_size=5,
        meta_batch_size=7,
        train_digits=list(range(7)),
        test_digits=list(range(7, 10)),        
        num_classes_per_batch=1,
        same_input_and_label=True)    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        latent_dim = 16
        # model = Vae(
        #     encoder_layers=[
        #         layers.Reshape((28, 28, 1)),
        #         layers.Conv2D(filters=32,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
        #         layers.BatchNormalization(trainable=True),
        #         layers.Activation('relu'),
        #         layers.Conv2D(filters=64,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
        #         layers.BatchNormalization(trainable=True),
        #         layers.Activation('relu'),
        #         layers.Conv2D(filters=128,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
        #         layers.BatchNormalization(trainable=True),
        #         layers.Activation('relu'),
        #         layers.Flatten(),
        #         layers.Dense(units=(latent_dim + latent_dim))
        #     ],
        #     decoder_layers=[
        #         layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
        #         layers.Reshape(target_shape=(7, 7, 32)),
        #         layers.Conv2DTranspose(filters=256,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
        #         layers.BatchNormalization(trainable=True),
        #         layers.Activation('relu'),
        #         layers.Conv2DTranspose(filters=128,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
        #         layers.BatchNormalization(trainable=True),
        #         layers.Activation('relu'),
        #         layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2, 2), padding="SAME", data_format="channels_last"),
        #         # layers.Activation('sigmoid'),
        #     ],
        #     latent_dim=latent_dim,
        #     sess=sess)
        model = Vae(
            encoder_layers=[
                layers.Reshape((32, 32, 3)),
                layers.Conv2D(filters=3,kernel_size=4,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Conv2D(filters=64,kernel_size=4,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Conv2D(filters=128,kernel_size=4,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=True),
                layers.Activation('relu'),
                layers.Flatten(),
                layers.Dense(units=(latent_dim + latent_dim))
            ],
            decoder_layers=[
                layers.Dense(units=4 * 4 * 128, activation=tf.nn.relu),
                layers.Reshape(target_shape=(4, 4, 128)),
                layers.Conv2DTranspose(filters=128,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
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
        # model = Vae(
        #     encoder_layers=[
        #         layers.Reshape((32, 32, 1)),
        #         layers.Conv2D(filters=32,kernel_size=3,strides=(2, 2),padding='same',activation="relu", data_format="channels_last"),
        #         layers.Conv2D(filters=64,kernel_size=3,strides=(2, 2),padding='same',activation="relu", data_format="channels_last"),
        #         layers.Conv2D(filters=128,kernel_size=3,strides=(2, 2),padding='same',activation="relu", data_format="channels_last"),
        #         layers.Flatten(),
        #         layers.Dense(units=(latent_dim + latent_dim))
        #     ],
        #     decoder_layers=[
        #         layers.Dense(units=4 * 4 * 64, activation=tf.nn.relu),
        #         layers.Reshape(target_shape=(4, 4, 64)),
        #         layers.Conv2DTranspose(filters=64,kernel_size=4,strides=(2, 2),padding="SAME",activation='relu', data_format="channels_last"),
        #         layers.Conv2DTranspose(filters=64,kernel_size=4,strides=(2, 2),padding="SAME",activation='relu', data_format="channels_last"),
        #         layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2, 2), padding="SAME", data_format="channels_last"),
        #     ],
        #     latent_dim=latent_dim,
        #     sess=sess)
      
        logger = Logger("maml_vae_cifar10", save_period=1000)

        maml = Maml(
            model,
            metasampler,
            sess,
            logger,
            compute_acc=False,
            num_updates=5,
            update_lr=0.001,
            meta_lr=0.0001)
         
        maml.train(
            train_itr=5000, 
            test_itr=1,
            test_interval=100,
            restore_model_path=None,
        )  
        
        # restore_model_path = logger._log_path
          
        # maml.test(
        #     test_itr=1,
        #     restore_model_path=restore_model_path
        # )
        
        
        
if __name__ == "__main__":
    main()
