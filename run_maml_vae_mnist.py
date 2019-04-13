#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Vae
import argparse

def main(args):
    digits = list(range(7))
    meta_batch_size = 7
    if args.test:
        digits = list(range(7, 10))
        meta_batch_size = 3

    metasampler = MnistMetaSampler(
        batch_size=1,
        meta_batch_size=meta_batch_size,
        digits=digits,        
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

        name = "maml_vae"
        num_updates = 5
        if args.test:
            name = "test_maml_vae"      
            num_updates=15      

        maml = Maml(
            model,
            metasampler,
            sess,
            compute_acc=False,
            num_updates=num_updates,
            update_lr=0.001,
            meta_lr=0.0001,
            name=name)

        if args.test:
            maml.test(1, args.restore_model_path)
        else:    
            maml.train(1000, args.restore_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('-m', dest='restore_model_path', action='store', required=False)

    args = parser.parse_args()
    if args.test and args.restore_model_path is None:
        parser.error("--test requires -m [restore model path]")

    main(args)
