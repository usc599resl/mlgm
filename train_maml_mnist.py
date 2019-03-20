#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from mlgm.algo import Maml
from mlgm.sampler import MnistSampler
from mlgm.model import Model


def main():
    tasks = [
        MnistSampler([3, 2, 7], batch_size=10),
        MnistSampler([1, 4, 8], batch_size=10),
        MnistSampler([5, 6, 9], batch_size=10)
    ]
    with tf.Session() as sess:
        model = Model([
            layers.Flatten(),
            layers.Dense(units=512, activation=tf.nn.relu),
            layers.Dropout(rate=0.2),
            layers.Dense(units=10, activation=tf.nn.softmax)
        ], {
            'shape': (None, 28, 28),
            'dtype': 'float32'
        }, {
            'shape': (None, ),
            'dtype': 'int64'
        }, sess)
        maml = Maml(model, tasks)
        maml.train(sess, n_itr=3)


if __name__ == "__main__":
    main()
