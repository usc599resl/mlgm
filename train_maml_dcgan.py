#!/usr/bin/env python
import tensorflow as tf

from mlgm.algo import MamlDcGan
from mlgm.model import DcGan
from mlgm.sampler import Cifar10MetaSampler

with tf.Session() as sess:
    sampler = Cifar10MetaSampler(
        batch_size=64,
        meta_batch_size=4,
        train_classes=list(range(0, 9)),
        test_classes=[9],
        num_classes_per_batch=1)
    dcgan = DcGan(sess, output_height=32, output_width=32)
    algo = MamlDcGan(
        dcgan,
        sampler,
        sess,
        name="maml_dcgan",
        num_updates=2,
        update_lr=1e-5,
        meta_lr=1e-5,
        metatrain_iterations=1000)
    algo.train()
