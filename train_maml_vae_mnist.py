#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Vae
import argparse

def main():
    metasampler = MnistMetaSampler(
        batch_size=1,
        meta_batch_size=7,
        train_digits=list(range(7)),
        test_digits=list(range(7, 10)),        
        num_classes_per_batch=1,
        same_input_and_label=True)
    with tf.Session() as sess:
        latent_dim = 8
        model = Vae(
            encoder_layers=[
                layers.Reshape((28, 28, 1)),
                layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='same',
                    activation="relu"),
                layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation="relu"),
                layers.Flatten(),
                layers.Dense(units=(latent_dim + latent_dim))
            ],
            decoder_layers=[
                layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ],
            latent_dim=latent_dim,
            sess=sess)
      

        maml = Maml(
            model,
            metasampler,
            sess,
            compute_acc=False,
            num_updates=5,
            update_lr=0.001,
            meta_lr=0.0001,
            name="maml_vae")
         
        maml.train(
            train_itr=2000, 
            test_itr=1,
            test_interval=100,
            restore_model_path=None
        )
        
        '''
        maml.test(
            test_itr=1,
            restore_model_path='./data/maml_vae/maml_vae_20_53_04_14_19/itr_950'
        )
        '''
        

if __name__ == "__main__":
    main()