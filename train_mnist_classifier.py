#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from mlgm.model import Model
from mlgm.sampler import MnistSampler

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
    # Only learn to classify digits from 0 to 6, so MAML can learn 7, 8 and 9
    # as tasks
    sampler = MnistSampler(training_digits=list(range(7)), batch_size=100)
    sess.run(tf.global_variables_initializer())
    num_epoch = 7
    for itr in range(num_epoch):
        end_of_epoch = False
        while not end_of_epoch:
            x_train, y_train, end_of_epoch = sampler.sample()
            model.optimize(x_train, y_train)
        x_test, y_test = sampler.get_test_set()
        it_acc = model.compute_acc(x_test, y_test)
        print("Epoch {}, accuracy {}".format(itr, it_acc))
    model.save_model()
