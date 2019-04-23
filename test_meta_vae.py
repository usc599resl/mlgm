#!/usr/bin/env python

import io

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from mlgm.algo import Maml
from mlgm.logger import Logger
from mlgm.model import Vae
from mlgm.sampler import MnistMetaSampler


def gen_fig(imgs_a, gen_imgs_a, sess):
    fig = plt.figure()
    for i, (img_a, gen_img_a) in enumerate(zip(imgs_a, gen_imgs_a)):
        plt.subplot(2, 3, (i + 1))
        plt.imshow(img_a[0], cmap='gray')
        plt.subplot(2, 3, 3 + (i + 1))
        plt.imshow(gen_img_a[0].reshape(28, 28), cmap='gray')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image("plot", image)
    return sess.run(summary_op)


def main():
    logger = Logger("test_meta_vae")
    metasampler = MnistMetaSampler(
        batch_size=1,
        meta_batch_size=3,
        train_digits=list(range(7, 10)),
        test_digits=list(range(7)),
        num_classes_per_batch=1)
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
        (input_a, label_a, input_b,
         label_b) = metasampler.build_inputs_and_labels()

        def task(inputs):
            logits = model.build_forward_pass(inputs)
            loss = model.build_loss(inputs, logits)
            return logits, loss

        out_dtype = (tf.float64, tf.float64)
        logits_sym, loss_sym = tf.map_fn(
            task,
            elems=(input_a),
            dtype=out_dtype,
            parallel_iterations=metasampler.meta_batch_size)
        metatrain_op = tf.train.AdamOptimizer(1e-3).minimize(loss_sym)

        sess.run(tf.global_variables_initializer())
        model.restore_model("data/maml_vae/maml_vae_21_23_04_06_19/itr_950")
        metasampler.restart_dataset(sess)

        for i in range(1000):
            try:
                imgs_a, gen_imgs_a, loss_num, _ = sess.run(
                    [input_a, logits_sym, loss_sym, metatrain_op])
                logger.new_summary()
                if not (i % 50):
                    tf_img = gen_fig(imgs_a, gen_imgs_a, sess)
                    logger.add_image(tf_img, i)
                logger.add_value("loss", loss_num.mean())
                logger.dump_summary(i)
            except tf.errors.OutOfRangeError:
                metasampler.restart_dataset(sess)
        logger.close()


if __name__ == "__main__":
    main()
