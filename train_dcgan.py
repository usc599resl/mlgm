#!/usr/bin/env python
import tensorflow as tf

from mlgm.logger import Logger
from mlgm.model import DcGan
from mlgm.sampler import Cifar10Sampler
from mlgm.utils import get_img_from_arr, transform_img

logger = Logger("DCGAN")
sampler = Cifar10Sampler(batch_size=64)

with tf.Session() as sess:
    train_in, train_lb = sampler.build_train_inputs_and_labels()

    dcgan = DcGan(sess, output_height=32, output_width=32)

    out = dcgan.build_forward_pass(train_in, training=True)
    g_out = out[0]
    d_loss, g_loss = dcgan.build_loss(train_lb, out)
    variables = dcgan.get_variables()
    d_optim = tf.train.AdamOptimizer(8e-5, 0.5).minimize(
        d_loss, var_list=variables["discriminator"])
    g_optim = tf.train.AdamOptimizer(8e-5, 0.5).minimize(
        g_loss, var_list=variables["generator"])

    sess.run(tf.global_variables_initializer())
    sampler.restart_train_dataset(sess)
    it = 0
    for i in range(6):
        while True:
            try:
                d_loss_t, _ = sess.run([d_loss, d_optim])
                g_loss_t, _ = sess.run([g_loss, g_optim])
                imgs, g_loss_t_1, _ = sess.run([g_out, g_loss, g_optim])
                imgs = transform_img(imgs)
                img_fig = get_img_from_arr(imgs)
                logger.new_summary()
                logger.add_value("d_loss", d_loss_t)
                logger.add_value("g_loss", g_loss_t)
                logger.add_value("g_loss_1", g_loss_t_1)
                logger.add_img("output_imgs", img_fig)
                logger.dump_summary(it)
                it += 1
            except tf.errors.OutOfRangeError:
                sampler.restart_train_dataset(sess)
                break
    logger.close()
