"""Simple MAML implementation.

Based on algorithm 1 from:
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning
for fast adaptation of deep networks." Proceedings of the 34th International
Conference on Machine Learning-Volume 70. JMLR. org, 2017.

https://arxiv.org/pdf/1703.03400.pdf
"""
import numpy as np
import tensorflow as tf

from mlgm.logger import Logger
from mlgm.utils import get_img_from_arr, transform_img


class MamlDcGan:
    def __init__(self,
                 model,
                 metasampler,
                 sess,
                 name="maml",
                 num_updates=1,
                 update_lr=0.0001,
                 meta_lr=0.0001,
                 pretrain_iterations=0,
                 metatrain_iterations=1000):
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._num_updates = num_updates
        self._update_lr = update_lr
        self._meta_lr = meta_lr
        self._pretrain_itr = pretrain_iterations
        self._metatrain_itr = metatrain_iterations
        self._logger = Logger(name)
        self._build_train()
        self._build_test()
        # self._logger.add_graph(self._sess.graph)

    def _build_train(self):
        with self._sess.graph.as_default():
            self._itr_in = tf.placeholder(dtype=tf.float32, shape=())
            input_a, label_a, input_b, label_b = \
                self._metasampler.build_train_inputs_and_labels()
            train_losses_a, output = self._build(
                input_a, label_a, self._metasampler.train_meta_batch_size,
                training=True)

            self._train_output = output

            self._d_train_losses_a = [i[0] for i in train_losses_a]
            self._g_train_losses_a = [i[1] for i in train_losses_a]
            dis_vars = self._model.get_variables()["discriminator"]
            gen_vars = self._model.get_variables()["generator"]
            with tf.variable_scope("pretrain", values=[train_losses_a]):
                self._dis_pretrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._d_train_losses_a[0], var_list=dis_vars)
                self._gen_pretrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._g_train_losses_a[0], var_list=gen_vars)

            with tf.variable_scope("metatrain", values=[train_losses_a]):
                self._dis_metatrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._d_train_losses_a[-1], var_list=dis_vars)
                self._gen_metatrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._g_train_losses_a[-1], var_list=gen_vars)

    def _build_test(self):
        with self._sess.graph.as_default():
            input_a, label_a, input_b, label_b = \
                    self._metasampler.build_test_inputs_and_labels()
            test_losses_a, output = self._build(
                input_a, label_a, self._metasampler.test_meta_batch_size)
            self._d_test_losses_a = [i[0] for i in test_losses_a]
            self._g_test_losses_a = [i[1] for i in test_losses_a]
            self._test_output = output

    def _build(self,
               dataset_in_a,
               dataset_lb_a,
               num_parallel_itr,
               training=False):
        def task_metalearn(args):
            input_a, label_a = args
            self._model.build_forward_pass(input_a, training=training)

            losses_a = []
            f_w = None
            outputs = None
            for i in range(self._num_updates):
                loss_a, f_w, outputs = self._build_update(
                    input_a, label_a, self._update_lr, f_w, training)
                losses_a.append((tf.math.reduce_mean(loss_a[0]),
                                 tf.math.reduce_mean(loss_a[1])))

            return losses_a, outputs

        out_dtype = ([(tf.float32, ) * 2] * self._num_updates, tf.float32)
        elems = (dataset_in_a, dataset_lb_a)
        losses_a, outputs_a = tf.map_fn(
            task_metalearn,
            elems=elems,
            dtype=out_dtype,
            parallel_iterations=num_parallel_itr)

        return losses_a, outputs_a

    def _build_update(self,
                      input_a,
                      label_a,
                      update_lr,
                      fast_weights=None,
                      training=False):
        values = [input_a, label_a, update_lr]
        with tf.variable_scope("update", values=values):
            output_a = self._model.build_forward_pass(input_a, fast_weights,
                                                      training)
            loss_a = self._model.build_loss(label_a, output_a)
            grads, weights = self._model.build_gradients(loss_a, fast_weights)
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                new_fast_weights = {
                    w: weights[w] - update_lr * grads[w]
                    for w in weights
                }
        return loss_a, new_fast_weights, output_a[0]

    def _compute_dis_pretrain(self):
        loss_a, _ = self._sess.run([self._d_train_losses_a[0],
            self._dis_pretrain_op])
        return loss_a

    def _compute_gen_pretrain(self):
        loss_a, _ = self._sess.run([self._g_train_losses_a[0],
            self._gen_pretrain_op])
        return loss_a

    def _compute_dis_metatrain(self, itr):
        losses_a, _ = self._sess.run([self._d_train_losses_a,
            self._dis_metatrain_op], {self._itr_in: itr})
        return losses_a

    def _compute_gen_metatrain(self, itr):
        losses_a, output, _ = self._sess.run([self._g_train_losses_a,
            self._train_output, self._gen_metatrain_op], {self._itr_in: itr})
        return losses_a, output

    def _test_metalearner(self):
        g_losses_a, d_losses_a, output = self._sess.run([self._g_test_losses_a,
            self._d_test_losses_a, self._test_output])
        return g_losses_a, d_losses_a, output

    def train(self, restore_model_path=None):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        self._metasampler.restart_train_dataset(self._sess)
        self._metasampler.restart_test_dataset(self._sess)
        for i in range(self._pretrain_itr + self._metatrain_itr):
            test_dis_losses_a = []
            test_gen_losses_a = []
            train_dis_losses_a = []
            train_gen_losses_a = []
            if i < self._pretrain_itr:
                self._metasampler.restart_train_dataset(self._sess)
                dis_loss_a = self._compute_dis_pretrain()
                self._metasampler.restart_train_dataset(self._sess)
                gen_loss_a = self._compute_gen_pretrain()
                self._logger.new_summary()
                self._logger.add_value("train_dis_loss_a", dis_loss_a.mean())
                self._logger.add_value("train_gen_loss_a", gen_loss_a.mean())
                self._logger.dump_summary(i)
            else:
                self._metasampler.restart_train_dataset(self._sess)
                dis_losses_a = self._compute_dis_metatrain(i)
                self._metasampler.restart_train_dataset(self._sess)
                gen_losses_a, train_outputs = self._compute_gen_metatrain(i)
                train_dis_losses_a.append(np.array(dis_losses_a).mean(axis=1))
                train_gen_losses_a.append(np.array(gen_losses_a).mean(axis=1))

                self._metasampler.restart_test_dataset(self._sess)
                g_losses_a, d_losses_a, test_output = self._test_metalearner()
                test_dis_losses_a.append(np.array(d_losses_a).mean(axis=1))
                test_gen_losses_a.append(np.array(g_losses_a).mean(axis=1))

                train_dis_losses_a = np.array(train_dis_losses_a).mean(axis=0)
                train_gen_losses_a = np.array(train_gen_losses_a).mean(axis=0)
                test_dis_losses_a = np.array(test_dis_losses_a).mean(axis=0)
                test_gen_losses_a = np.array(test_gen_losses_a).mean(axis=0)
                self._logger.new_summary()
                self._logger.add_value("train_dis_losses_a/update_",
                                       train_dis_losses_a.tolist())
                self._logger.add_value("train_gen_losses_a/update_",
                                       train_gen_losses_a.tolist())
                self._logger.add_value("test_dis_losses_a/update_",
                                       test_dis_losses_a.tolist())
                self._logger.add_value("test_gen_losses_a/update_",
                                       test_gen_losses_a.tolist())

                if i % 100:
                    for j, out in enumerate(train_outputs):
                        imgs = transform_img(out)
                        img_fig = get_img_from_arr(imgs)
                        self._logger.add_img(
                                "train_output_".format(j), img_fig)

                    for j, out in enumerate(test_output):
                        imgs = transform_img(out)
                        img_fig = get_img_from_arr(imgs)
                        self._logger.add_img(
                                "test_output_".format(j), img_fig)

                self._logger.dump_summary(i)
        self._logger.save_tf_variables(self._model.get_variables(), i,
                                       self._sess)
        self._logger.close()
