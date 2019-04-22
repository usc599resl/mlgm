#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler.omniglot_meta_sampler import OmniglotMetaSampler
from mlgm.model import Vae
from mlgm.logger import Logger

def main():
    metasampler = OmniglotMetaSampler(
        batch_size=5,
        meta_batch_size=7,
        num_of_class=10,
        num_classes_per_batch=1,
        same_input_and_label=True)
    with tf.Session() as sess:
        latent_dim = 16
        model = Vae(
            encoder_layers=[
                layers.Reshape((28, 28, 1)),
                layers.Conv2D(filters=32,kernel_size=3,strides=(2, 2),padding='same',activation="relu"),
                layers.Conv2D(filters=64,kernel_size=3,strides=(2, 2),padding='same',activation="relu"),
                layers.Conv2D(filters=128,kernel_size=3,strides=(2, 2),padding='same',activation="relu"),
                layers.Flatten(),
                layers.Dense(units=(latent_dim + latent_dim))
            ],
            decoder_layers=[
                layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=(2, 2),padding="SAME",activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=(2, 2),padding="SAME",activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ],
            latent_dim=latent_dim,
            sess=sess)
      
        logger = Logger("maml_vae_omniglot", save_period=2000)

        maml = Maml(
            model,
            metasampler,
            sess,
            logger,
            compute_acc=False,
            num_updates=1,
            update_lr=0.001,
            meta_lr=0.0001)
         
        maml.train(
            train_itr=2000, 
            test_itr=1,
            test_interval=100,
            restore_model_path=None
        )        
  
        # maml.test(
        #     test_itr=1,
        #     restore_model_path='./data/maml_vae_omniglot/maml_vae_omniglot_latest/itr_2000'
        # )
        
        
if __name__ == "__main__":
    main()
