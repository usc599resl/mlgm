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
                 metasampler,
                 sess,
                 name="maml",
                 num_updates=1,
                 update_lr=0.9,
                 beta=0.9,
                 pre_train_iterations=1000,
                 metatrain_iterations=1000):
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._num_updates = num_updates
        self._update_lr = update_lr
        self._beta = beta
        self._pre_train_itr = pre_train_iterations
        self._metatrain_itr = metatrain_iterations
        self._logger = Logger(name)
        self._build()
        self._logger.add_graph(self._sess.graph)

    def _build(self):
        with self._sess.graph.as_default():
            # Algorithm Inputs
            (self._input_a, self._label_a, self._input_b,
             self._label_b) = self._metasampler.build_inputs_and_labels()

            # This model builds the weights and the accuracy
            def task_metalearn(args):
                input_a, label_a, input_b, label_b = args
                output = self._model.build_forward_pass(input_a)
                acc = self._model.build_accuracy(label_a, output)

                # loss_a is only used for pre training
                loss_a = None
                acc_a = None
                losses_b = []
                accs_b = []
                f_w = None
                for i in range(self._num_updates):
                    loss, acc, loss_b, acc_b, f_w = self._build_update(
                        input_a, label_a, input_b, label_b, self._update_lr,
                        f_w)
                    if loss_a is None:
                        loss_a = tf.math.reduce_mean(loss)
                        acc_a = acc
                    losses_b.append(tf.math.reduce_mean(loss_b))
                    accs_b.append(acc_b)

                return loss_a, acc_a, losses_b, accs_b

            out_dtype = (tf.float64, tf.float32,
                         [tf.float64] * self._num_updates,
                         [tf.float32] * self._num_updates)
            self._loss_a, self._acc_a, self._losses_b, self._accs_b = tf.map_fn(
                task_metalearn,
                elems=(self._input_a, self._label_a, self._input_b,
                       self._label_b),
                dtype=out_dtype,
                parallel_iterations=self._metasampler.meta_batch_size)

            with tf.variable_scope("pretrain", values=[self._loss_a]):
                self._pretrain_op = tf.train.AdamOptimizer().minimize(
                    self._loss_a)

            if self._metatrain_itr > 0:
                with tf.variable_scope("metatrain", values=[self._losses_b]):
                    self._metatrain_op = tf.train.AdamOptimizer().minimize(
                        self._losses_b[self._num_updates - 1])

    def _build_update(self,
                      input_a,
                      label_a,
                      input_b,
                      label_b,
                      update_lr,
                      fast_weights=None):
        values = [input_a, label_a, input_b, label_b, update_lr]
        loss_a = None
        loss_b = None
        with tf.variable_scope("update", values=values):
            output_a = self._model.build_forward_pass(input_a, fast_weights)
            label_a_oh = tf.one_hot(label_a, depth=10)
            loss_a = self._model.build_loss(label_a_oh, output_a)
            acc_a = self._model.build_accuracy(label_a, output_a)
            grads, weights = self._model.build_gradients(loss_a, fast_weights)
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                new_fast_weights = {
                    w: weights[w] - update_lr * grads[w]
                    for w, g in zip(weights, grads)
                }
            output_b = self._model.build_forward_pass(input_b,
                                                      new_fast_weights)
            label_b_oh = tf.one_hot(label_b, depth=10)
            loss_b = self._model.build_loss(label_b_oh, output_b)
            acc_b = self._model.build_accuracy(label_b, output_b)
        return loss_a, acc_a, loss_b, acc_b, new_fast_weights

    def _compute_metatrain(self):
        loss_a, acc_a, losses_b, accs_b, _ = self._sess.run([
            self._loss_a, self._acc_a, self._losses_b, self._accs_b,
            self._metatrain_op
        ])
        return loss_a, acc_a, losses_b, accs_b

    def _compute_accuracy(self, input_vals, labels):
        feed_dict = {self._input_a: input_vals, self._label_a: labels}
        return self._sess.run(self._acc, feed_dict=feed_dict)

    def train(self, restore_model_path=None):
        self._sess.run(tf.global_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        self._metasampler.restart_dataset(self._sess)
        for i in range(self._pre_train_itr + self._metatrain_itr):
            try:
                loss_a, acc_a, losses_b, accs_b = self._compute_metatrain()
                loss_a = np.mean(loss_a)
                acc_a = np.mean(acc_a)
                losses_b = np.array(losses_b)
                losses_b = np.mean(losses_b, axis=1)
                accs_b = np.array(accs_b).mean(axis=1)
                self._logger.new_summary()
                self._logger.add_value("loss_a", loss_a)
                self._logger.add_value("loss_b/update_", losses_b.tolist())
                self._logger.add_value("acc_a", acc_a)
                self._logger.add_value("acc_b/update_", accs_b.tolist())
                self._logger.dump_summary(i)
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_dataset(self._sess)
        self._logger.close()
