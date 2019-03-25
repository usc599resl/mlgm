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


class Maml:
    def __init__(self,
                 model,
                 tasks,
                 sess,
                 name="maml",
                 num_updates=1,
                 alpha=1.0,
                 beta=0.9,
                 pre_train_iterations=1000,
                 metatrain_iterations=1000):
        self._model = model
        self._tasks = tasks
        self._sess = sess
        self._num_updates = num_updates
        self._alpha = alpha
        self._beta = beta
        self._pre_train_itr = pre_train_iterations
        self._metatrain_itr = metatrain_iterations
        self._logger = Logger(name)
        self._build()
        self._logger.add_graph(self._sess.graph)

    def _build(self):
        with self._sess.graph.as_default():
            # Algorithm Inputs
            self._input_a = self._tasks[0].build_input_placeholder(
                name="input_a")
            self._label_a = self._tasks[0].build_label_placeholder(
                name="label_a", dtype=tf.dtypes.int64)
            self._input_b = self._tasks[0].build_input_placeholder(
                name="input_b")
            self._label_b = self._tasks[0].build_label_placeholder(
                name="label_b", dtype=tf.dtypes.int64)
            alpha = tf.constant(
                self._alpha, name="alpha", dtype=self._input_a.dtype)
            # This model builds the weights and the accuracy
            output = self._model.build_forward_pass(self._input_a)
            self._acc = self._model.build_accuracy(self._label_a, output)

            # loss_a is only used for pre training
            self._loss_a = None
            self._loss_b = None
            for i in range(self._num_updates):
                loss_a, loss_b = self._build_update(
                    self._input_a, self._label_a, self._input_b, self._label_b,
                    alpha)
                if self._loss_a is None:
                    self._loss_a = loss_a
            self._loss_b = loss_b

            with tf.variable_scope("pretrain", values=[self._loss_a]):
                self._pretrain_op = tf.train.AdamOptimizer().minimize(
                    self._loss_a)

            if self._metatrain_itr > 0:
                with tf.variable_scope("metatrain", values=[self._loss_b]):
                    self._metatrain_op = tf.train.AdamOptimizer().minimize(
                        self._loss_b)

    def _build_update(self, input_a, label_a, input_b, label_b, alpha):
        values = [input_a, label_a, input_b, label_b, alpha]
        loss_a = None
        loss_b = None
        with tf.variable_scope("update", values=values):
            output_a = self._model.build_forward_pass(input_a)
            loss_a = self._model.build_loss(label_a, output_a)
            grads, weights = self._model.build_compute_gradients(loss_a)
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                fast_weights = {
                    w.name: w - alpha * g
                    for w, g in zip(weights, grads)
                }
            self._model.assign_model_params(fast_weights)
            output_b = self._model.build_forward_pass(input_b)
            loss_b = self._model.build_loss(label_b, output_b)
        return loss_a, loss_b

    def _compute_pretrain(self, input_vals, labels):
        feed_dict = {self._input_a: input_vals, self._label_a: labels}
        loss, acc, _ = self._sess.run(
            [self._loss_a, self._acc, self._pretrain_op], feed_dict=feed_dict)
        return loss, acc

    def _compute_accuracy(self, input_vals, labels):
        feed_dict = {self._input_a: input_vals, self._label_a: labels}
        return self._sess.run(self._acc, feed_dict=feed_dict)

    def train(self, restore_model_path=None):
        self._sess.run(tf.global_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        for i in range(self._pre_train_itr + self._metatrain_itr):
            done = False
            pretrain_losses = []
            pretrain_accs = []
            while not done:
                input_a, label_a, done = self._tasks[0].sample()
                pretrain_loss, pretrain_acc = self._compute_pretrain(
                    input_a, label_a)
                pretrain_losses.append(pretrain_loss)
                pretrain_accs.append(pretrain_acc)
            pretrain_loss = np.mean(pretrain_losses)
            pretrain_acc = np.mean(pretrain_accs)
            input_test, label_test = self._tasks[0].get_test_set()
            test_acc = self._compute_accuracy(input_test, label_test)
            self._logger.new_summary()
            self._logger.add_value("pretrain_loss", pretrain_loss)
            self._logger.add_value("accuracy/pretrain_acc", pretrain_acc)
            self._logger.add_value("accuracy/test_acc", test_acc)
            self._logger.dump_summary(i)
        self._logger.close()
