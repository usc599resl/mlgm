#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from mlgm.model import Model
from mlgm.logger import Logger
from mlgm.sampler import MnistSampler

with tf.Session() as sess:
    logger = Logger("mnist_classifier", std_out_period=1, save_period=1)
    model = Model([
        layers.Flatten(),
        layers.Dense(units=512, activation=tf.nn.relu),
        layers.Dropout(rate=0.2),
        layers.Dense(units=10, activation=tf.nn.softmax)
    ], sess, {
        'shape': (None, 28, 28),
        'dtype': 'float32'
    }, {
        'shape': (None, 10),
        'dtype': 'int64'
    })
    model.build()
    # Only learn to classify digits from 0 to 6, so MAML can learn 7, 8 and 9
    # as tasks
    sampler = MnistSampler(training_digits=list(range(7)), batch_size=100,
            one_hot_labels=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    num_epoch = 7
    for itr in range(num_epoch):
        end_of_epoch = False
        while not end_of_epoch:
            x_train, y_train, end_of_epoch = sampler.sample()
            model.optimize(x_train, y_train)
        x_test, y_test = sampler.get_test_set()
        it_acc = model.compute_acc(x_test, y_test)
        logger.new_summary()
        logger.add_value("accuracy", it_acc)
        logger.dump_summary(itr)
    logger.save_tf_variables(model.get_variables(), itr, sess)
