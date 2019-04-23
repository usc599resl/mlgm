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
            ],
            latent_dim=latent_dim,
            sess=sess)
      
        logger = Logger("maml_vae_cifar10", save_period=2499)

        maml = Maml(
            model,
            metasampler,
            sess,
            logger,
            compute_acc=False,
            num_updates=1,
            update_lr=0.0005,
            meta_lr=0.00005)
     
        maml.train(
            train_itr=10000, 
            test_itr=1,
            test_interval=100,
            # restore_model_path='./data/maml_vae_cifar10/maml_vae_cifar10_2019_04_23_00_40_22_233440_/itr_7497',
            restore_model_path=None,
        ) 
                  
        # maml.test(
        #     test_itr=1,
        #     restore_model_path='./data/maml_vae_cifar10/maml_vae_cifar10_2019_04_23_00_10_56_512339_/itr_4998',
        # )
        
if __name__ == "__main__":
    main()
