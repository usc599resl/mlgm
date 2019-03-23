#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from mlgm.algo import Maml
from mlgm.sampler import MnistSampler
from mlgm.model import Model


def main():
    known_digits = list(range(7))
    tasks = [
        MnistSampler(known_digits + [7], batch_size=10),
        MnistSampler(known_digits + [8], batch_size=10),
        MnistSampler(known_digits + [9], batch_size=10)
    ]
    with tf.Session() as sess:
        model = Model([
            layers.Flatten(),
            layers.Dense(units=512, activation=tf.nn.relu),
            layers.Dropout(rate=0.2),
            layers.Dense(units=10, activation=tf.nn.softmax)
        ], {
            'shape': (None, 28, 28),
            'dtype': 'float32',
            'name': 'Input'
        }, {
            'shape': (None, ),
            'dtype': 'int64',
            'name': 'Label'
        }, sess)
        # import pdb
        # pdb.set_trace()
        maml = Maml(model, tasks)
        # model.restore_model("data/model_12_44_03_20_19/model")
        maml.train(sess, n_itr=3)


if __name__ == "__main__":
    main()
