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
            train_loss_a, train_losses_b, output = self._build(
                input_a,
                label_a,
                input_b,
                label_b,
                self._metasampler.train_meta_batch_size,
                training=True)

            self._output = output

            dis_vars = self._model.get_variables()["discriminator"]
            gen_vars = self._model.get_variables()["generator"]
            with tf.variable_scope("pretrain", values=[train_loss_a]):
                self._d_train_loss_a = train_loss_a[0]
                self._g_train_loss_a = train_loss_a[1]
                self._dis_pretrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._d_train_loss_a, var_list=dis_vars)
                self._gen_pretrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._g_train_loss_a, var_list=gen_vars)

            with tf.variable_scope("metatrain", values=[train_losses_b]):
                self._d_train_losses_b = [i[0] for i in train_losses_b]
                self._g_train_losses_b = [i[1] for i in train_losses_b]
                self._dis_metatrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._d_train_losses_b[-1], var_list=dis_vars)
                self._gen_metatrain_op = tf.train.AdamOptimizer(
                    self._meta_lr, 0.5).minimize(
                        self._g_train_losses_b[-1], var_list=gen_vars)

    def _build_test(self):
        with self._sess.graph.as_default():
            input_a, label_a, input_b, label_b = \
                    self._metasampler.build_test_inputs_and_labels()
            test_loss_a, test_losses_b, output = self._build(
                input_a, label_a, input_b, label_b,
                self._metasampler.test_meta_batch_size)
            self._d_test_loss_a = test_loss_a[0]
            self._g_test_loss_a = test_loss_a[1]
            self._d_test_losses_b = [i[0] for i in test_losses_b]
            self._g_test_losses_b = [i[1] for i in test_losses_b]

    def _build(self,
               dataset_in_a,
               dataset_lb_a,
               dataset_in_b,
               dataset_lb_b,
               num_parallel_itr,
               training=False):
        def task_metalearn(args):
            input_a, label_a, input_b, label_b = args
            self._model.build_forward_pass(input_a, training=training)

            loss_a = None
            losses_b = []
            f_w = None
            output_b = None
            for i in range(self._num_updates):
                loss, loss_b, f_w, output_b = self._build_update(
                    input_a, label_a, input_b, label_b, self._update_lr, f_w,
                    training)
                if loss_a is None:
                    loss_a = (tf.math.reduce_mean(loss[0]),
                              tf.math.reduce_mean(loss[1]))
                losses_b.append((tf.math.reduce_mean(loss_b[0]),
                                 tf.math.reduce_mean(loss_b[1])))

            return loss_a, losses_b, output_b[0]

        out_dtype = ((tf.float32, ) * 2,
                     [(tf.float32, ) * 2] * self._num_updates, tf.float32)
        elems = (dataset_in_a, dataset_lb_a, dataset_in_b, dataset_lb_b)
        loss_a, losses_b, output_b = tf.map_fn(
            task_metalearn,
            elems=elems,
            dtype=out_dtype,
            parallel_iterations=num_parallel_itr)

        return loss_a, losses_b, output_b

    def _build_update(self,
                      input_a,
                      label_a,
                      input_b,
                      label_b,
                      update_lr,
                      fast_weights=None,
                      training=False):
        values = [input_a, label_a, input_b, label_b, update_lr]
        loss_a = None
        loss_b = None
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
            output_b = self._model.build_forward_pass(
                input_b, new_fast_weights, training)
            loss_b = self._model.build_loss(label_b, output_b)
        return loss_a, loss_b, new_fast_weights, output_b

    def _compute_dis_pretrain(self):
        loss_a, _ = self._sess.run([self._d_train_loss_a,
            self._dis_pretrain_op])
        return loss_a

    def _compute_gen_pretrain(self):
        loss_a, _ = self._sess.run([self._g_train_loss_a,
            self._gen_pretrain_op])
        return loss_a

    def _compute_dis_metatrain(self, itr):
        loss_a, losses_b, _ = self._sess.run([
            self._d_train_loss_a, self._d_train_losses_b,
            self._dis_metatrain_op
        ], {self._itr_in: itr})
        return loss_a, losses_b

    def _compute_gen_metatrain(self, itr):
        loss_a, losses_b, output, _ = self._sess.run([
            self._g_train_loss_a, self._g_train_losses_b, self._output,
            self._gen_metatrain_op
        ], {self._itr_in: itr})
        return loss_a, losses_b, output

    def _test_prelearner(self):
        loss_a = self._sess.run([self._test_loss_a])
        return loss_a

    def _test_metalearner(self):
        g_loss_a, g_losses_b, d_loss_a, d_losses_b = self._sess.run([
            self._g_test_loss_a, self._g_test_losses_b, self._d_test_loss_a,
            self._d_test_losses_b
        ])
        return g_loss_a, g_losses_b, d_loss_a, d_losses_b

    def train(self, restore_model_path=None):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        self._metasampler.restart_train_dataset(self._sess)
        self._metasampler.restart_test_dataset(self._sess)
        for i in range(self._pretrain_itr + self._metatrain_itr):
            test_dis_loss_a = []
            test_dis_losses_b = []
            test_gen_loss_a = []
            test_gen_losses_b = []
            train_dis_loss_a = []
            train_dis_losses_b = []
            train_gen_loss_a = []
            train_gen_losses_b = []
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
                dis_loss_a, dis_losses_b = \
                    self._compute_dis_metatrain(i)
                self._metasampler.restart_train_dataset(self._sess)
                gen_loss_a, gen_losses_b, output = \
                    self._compute_gen_metatrain(i)
                train_dis_loss_a.append(np.mean(dis_loss_a))
                train_dis_losses_b.append(np.array(dis_losses_b).mean(axis=1))
                train_gen_loss_a.append(np.mean(gen_loss_a))
                train_gen_losses_b.append(np.array(gen_losses_b).mean(axis=1))

                self._metasampler.restart_test_dataset(self._sess)
                g_loss_a, g_losses_b, d_loss_a, d_losses_b = \
                    self._test_metalearner()
                test_dis_loss_a.append(np.mean(d_loss_a))
                test_dis_losses_b.append(np.array(d_losses_b).mean(axis=1))
                test_gen_loss_a.append(np.mean(g_loss_a))
                test_gen_losses_b.append(np.array(g_losses_b).mean(axis=1))

                train_dis_loss_a = np.array(train_dis_loss_a).mean()
                train_dis_losses_b = np.array(train_dis_losses_b).mean(axis=0)
                train_gen_loss_a = np.array(train_gen_loss_a).mean()
                train_gen_losses_b = np.array(train_gen_losses_b).mean(axis=0)
                test_dis_loss_a = np.array(test_dis_loss_a).mean()
                test_dis_losses_b = np.array(test_dis_losses_b).mean(axis=0)
                test_gen_loss_a = np.array(test_gen_loss_a).mean()
                test_gen_losses_b = np.array(test_gen_losses_b).mean(axis=0)
                self._logger.new_summary()
                self._logger.add_value("train_dis_loss_a", train_dis_loss_a)
                self._logger.add_value("train_dis_losses_b/update_",
                                       train_dis_losses_b.tolist())
                self._logger.add_value("train_gen_loss_a", train_gen_loss_a)
                self._logger.add_value("train_gen_losses_b/update_",
                                       train_gen_losses_b.tolist())
                self._logger.add_value("test_dis_loss_a", test_dis_loss_a)
                self._logger.add_value("test_dis_losses_b/update_",
                                       test_dis_losses_b.tolist())
                self._logger.add_value("test_gen_loss_a", test_gen_loss_a)
                self._logger.add_value("test_gen_losses_b/update_",
                                       test_gen_losses_b.tolist())
                imgs = transform_img(output[0])
                img_fig = get_img_from_arr(imgs)
                self._logger.add_img("output_mnist", img_fig)
                self._logger.dump_summary(i)
        self._logger.save_tf_variables(self._model.get_variables(), i,
                                       self._sess)
        self._logger.close()
