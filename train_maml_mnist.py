#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from mlgm.algo import Maml
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Model


def main():
    metasampler = MnistMetaSampler(
        batch_size=4,
        meta_batch_size=4,
        train_digits=list(range(7)),
        test_digits=list(range(7, 10)),
        num_classes_per_batch=3)
    with tf.Session() as sess:
        # DO NOT USE Dropout as a layer:
        # https://github.com/keras-team/keras/issues/9288
        model = Model([
            layers.Flatten(),
            layers.Dense(units=512, activation=tf.nn.relu),
            layers.Dense(units=10, activation=tf.nn.softmax)
        ], sess)
        maml = Maml(
            model,
            metasampler,
            sess,
            num_updates=3,
            update_lr=0.1,
            pre_train_iterations=0,
            metatrain_iterations=1000)
        maml.train()


if __name__ == "__main__":
    main()
