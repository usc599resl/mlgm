#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
from mlgm.layers import Dropout

from mlgm.algo import Maml
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Model
from mlgm.logger import Logger

def main():
    metasampler = MnistMetaSampler(
        batch_size=4,
        meta_batch_size=4,
        train_digits=list(range(7)),
        test_digits=list(range(7, 10)),
        num_classes_per_batch=3,
        one_hot_labels=True)
    with tf.Session() as sess:
        model = Model([
            layers.Flatten(),
            layers.Dense(units=512, activation=tf.nn.relu),
            Dropout(0.2),
            layers.Dense(units=10, activation=tf.nn.softmax)
        ], sess)

        logger = Logger("maml_mnist_classifier", save_period=500)
        
        maml = Maml(
            model,
            metasampler,
            sess,
            logger,
            num_updates=3,
            update_lr=0.01,
            meta_lr=0.0005)
        
        maml.train(
            train_itr=1000, 
            test_itr=1,
            test_interval=100,
            restore_model_path=None
        )

        '''
        maml.test(
            test_itr=1,
            restore_model_path='./data/maml_mnist/maml_mnist_17_04_04_20_19/itr_900',
            log_images=False
        )
        '''


if __name__ == "__main__":
    main()
