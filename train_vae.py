#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from mlgm.logger import Logger
from mlgm.model import Vae
from mlgm.sampler import MnistMetaSampler

metasampler = MnistMetaSampler(
    batch_size=2,
    meta_batch_size=7,
    train_digits=list(range(7)),
    test_digits=list(range(7, 10)),
    num_classes_per_batch=1)
logger = Logger("vae")

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
            layers.Dense(units=7*7*32, activation=tf.nn.relu),
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
    input_a, label_a, input_b, label_b = metasampler.build_inputs_and_labels()

    def task(inputs):
        logits = model.build_forward_pass(inputs)
        loss = model.build_loss(inputs, logits)
        return loss

    out_dtype = (tf.float64)
    loss_sym = tf.map_fn(
        task,
        elems=(input_a),
        dtype=out_dtype,
        parallel_iterations=metasampler.meta_batch_size)
    metatrain_op = tf.train.AdamOptimizer(5e-3).minimize(loss_sym)

    sess.run(tf.global_variables_initializer())
    metasampler.restart_dataset(sess)
    for i in range(1000):
        try:
            loss_num, _ = sess.run([loss_sym, metatrain_op])
            logger.new_summary()
            logger.add_value("loss", loss_num.mean())
            logger.dump_summary(i)
        except tf.errors.OutOfRangeError:
            metasampler.restart_dataset(sess)
    logger.close()
