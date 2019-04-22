#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler import Cifar10MetaSampler
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
    with tf.Session() as sess:
        latent_dim = 16
        model = Vae(
            encoder_layers=[
                layers.Reshape((32, 32, 3)),
                layers.Conv2D(filters=32,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=False),
                layers.Activation('relu'),
                layers.Conv2D(filters=64,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=False),
                layers.Activation('relu'),
                layers.Conv2D(filters=128,kernel_size=3,strides=(2, 2),padding='same', data_format="channels_last"),
                layers.BatchNormalization(trainable=False),
                layers.Activation('relu'),
                layers.Flatten(),
                layers.Dense(units=(latent_dim + latent_dim))
            ],
            decoder_layers=[
                layers.Dense(units=4 * 4 * 64, activation=tf.nn.relu),
                layers.Reshape(target_shape=(4, 4, 64)),
                layers.Conv2DTranspose(filters=64,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
                layers.BatchNormalization(trainable=False),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=64,kernel_size=4,strides=(2, 2),padding="SAME", data_format="channels_last"),
                layers.BatchNormalization(trainable=False),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="SAME", data_format="channels_last"),
                layers.Activation('sigmoid'),
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
      
        logger = Logger("maml_vae_cifar10", save_period=1900)

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
            train_itr=2000, 
            test_itr=1,
            test_interval=100,
            restore_model_path=None,
        )  
        
        '''             
        maml.test(
            test_itr=1,
            restore_model_path='./data/maml_vae/maml_vae_22_42_04_16_19/itr_1950'
        )
        '''
        
        
if __name__ == "__main__":
    main()
