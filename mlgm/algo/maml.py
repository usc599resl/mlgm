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
from mlgm.utils import get_img_from_arr


class Maml:
    def __init__(self,
                 model,
                 metasampler,
                 sess,
                 compute_acc=True,
                 name="maml",
                 num_updates=1,
                 update_lr=0.0001,
                 meta_lr=0.0001,
                 metatrain_iterations=1000):
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._compute_acc = compute_acc
        self._num_updates = num_updates
        self._update_lr = update_lr
        self._meta_lr = meta_lr
        self._metatrain_itr = metatrain_iterations
        self._logger = Logger(name)
        self._build_train()
        self._build_test()
        # self._logger.add_graph(self._sess.graph)

    def _build_train(self):
        with self._sess.graph.as_default():
            input_a, label_a, input_b, label_b = \
                    self._metasampler.build_train_inputs_and_labels()
            (self._train_loss_a, self._train_acc_a, self._train_losses_b,
                    self._train_accs_b, _) = self._build(
                            input_a, label_a, input_b, label_b,
                            self._metasampler.train_meta_batch_size)

            with tf.variable_scope("metatrain", values=[self._train_losses_b]):
                self._metatrain_op = tf.train.AdamOptimizer(
                    self._meta_lr).minimize(
                        self._train_losses_b[self._num_updates - 1])

    def _build_test(self):
        with self._sess.graph.as_default():
            input_a, label_a, input_b, label_b = \
                    self._metasampler.build_test_inputs_and_labels()
            (self._test_loss_a, self._test_acc_a, self._test_losses_b,
                    self._test_accs_b, self._output) = self._build(
                            input_a, label_a, input_b, label_b,
                            self._metasampler.test_meta_batch_size)
            self._input_b = input_b

    def _build(self, dataset_in_a, dataset_lb_a, dataset_in_b, dataset_lb_b,
            num_parallel_itr):

        def task_metalearn(args):
            input_a, label_a, input_b, label_b = args
            output = self._model.build_forward_pass(input_a)
            acc = self._model.build_accuracy(label_a, output)

            loss_a = None
            acc_a = None
            losses_b = []
            accs_b = []
            f_w = None
            output_b = None
            for i in range(self._num_updates):
                loss, acc, loss_b, acc_b, f_w, output_b = self._build_update(
                    input_a, label_a, input_b, label_b, self._update_lr,
                    f_w)
                if loss_a is None:
                    loss_a = tf.math.reduce_mean(loss)
                    acc_a = acc
                losses_b.append(tf.math.reduce_mean(loss_b))
                accs_b.append(acc_b)

            return loss_a, acc_a, losses_b, accs_b, output_b

        out_dtype = (tf.float32, tf.float32,
                     [tf.float32] * self._num_updates,
                     [tf.float32] * self._num_updates, tf.float32)
        elems = (dataset_in_a, dataset_lb_a, dataset_in_b, dataset_lb_b)
        loss_a, acc_a, losses_b, accs_b, output_b = tf.map_fn(
             task_metalearn,
             elems=elems,
             dtype=out_dtype,
             parallel_iterations=num_parallel_itr)

        return loss_a, acc_a, losses_b, accs_b, output_b

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
            loss_a = self._model.build_loss(label_a, output_a)
            acc_a = self._model.build_accuracy(label_a, output_a)
            grads, weights = self._model.build_gradients(loss_a, fast_weights)
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                new_fast_weights = {
                    w: weights[w] - update_lr * grads[w]
                    for w in weights
                }
            output_b = self._model.build_forward_pass(input_b,
                                                      new_fast_weights)
            loss_b = self._model.build_loss(label_b, output_b)
            acc_b = self._model.build_accuracy(label_b, output_b)
        return loss_a, acc_a, loss_b, acc_b, new_fast_weights, output_b

    def _compute_metatrain(self, acc=True):
        if acc:
            loss_a, acc_a, losses_b, accs_b, _ = self._sess.run([
                self._train_loss_a, self._train_acc_a, self._train_losses_b,
                self._train_accs_b, self._metatrain_op
            ])
            return loss_a, acc_a, losses_b, accs_b
        else:
            loss_a, losses_b, _ = self._sess.run(
                [self._train_loss_a, self._train_losses_b, self._metatrain_op])
            return loss_a, losses_b

    def _test_metalearner(self, acc=True):
        if acc:
            loss_a, acc_a, losses_b, accs_b, output, input_b = self._sess.run([
                self._test_loss_a, self._test_acc_a, self._test_losses_b,
                self._test_accs_b, self._output, self._input_b
            ])
            return loss_a, acc_a, losses_b, accs_b, output, input_b
        else:
            loss_a, losses_b, output, input_b = self._sess.run(
                [self._test_loss_a, self._test_losses_b, self._output,
                    self._input_b])
            return loss_a, losses_b, output, input_b

    def train(self, restore_model_path=None):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        self._metasampler.restart_train_dataset(self._sess)
        self._metasampler.restart_test_dataset(self._sess)
        for i in range(self._metatrain_itr):
            test_losses_a = []
            test_losses_b = []
            train_losses_a = []
            train_losses_b = []
            test_accs_a = []
            test_accs_b = []
            train_accs_a = []
            train_accs_b = []
            while True:
                try:
                    if self._compute_acc:
                        (loss_a, acc_a, losses_b,
                         accs_b) = self._compute_metatrain(
                                 acc=self._compute_acc)
                        train_accs_a.append(acc_a)
                        train_accs_b.append(np.array(accs_b).mean(axis=1))
                    else:
                        loss_a, losses_b = self._compute_metatrain(
                                acc=self._compute_acc)
                    train_losses_a.append(np.mean(loss_a))
                    train_losses_b.append(np.array(losses_b).mean(axis=1))
                except tf.errors.OutOfRangeError:
                    self._metasampler.restart_train_dataset(self._sess)
                    break

            while True:
                try:
                    if self._compute_acc:
                        (loss_a, acc_a, losses_b,
                         accs_b, out, in_b) = self._test_metalearner(
                                 acc=self._compute_acc)
                        test_accs_a.append(acc_a)
                        test_accs_b.append(np.array(accs_b).mean(axis=1))
                    else:
                        loss_a, losses_b, out, in_b = self._test_metalearner(
                                acc=self._compute_acc)
                    test_losses_a.append(np.mean(loss_a))
                    test_losses_b.append(
                            np.array(losses_b).mean(axis=1))
                except tf.errors.OutOfRangeError:
                    self._metasampler.restart_test_dataset(self._sess)
                    break

            train_loss_a = np.array(train_losses_a).mean()
            train_losses_b = np.array(train_losses_b).mean(axis=0)
            test_loss_a = np.array(test_losses_a).mean()
            test_losses_b = np.array(test_losses_b).mean(axis=0)
            self._logger.new_summary()
            self._logger.add_value("train_loss_a", train_loss_a)
            self._logger.add_value(
                    "train_loss_b/update_", train_losses_b.tolist())
            self._logger.add_value("test_loss_a", test_loss_a)
            self._logger.add_value(
                    "test_loss_b/update_", test_losses_b.tolist())
            out = out.reshape(out.shape[:-1])
            img_arr = np.append(out, in_b, axis=1)
            encoded_img = get_img_from_arr(img_arr)
            self._logger.add_img("output_mnist", encoded_img)
            if self._compute_acc:
                train_acc_a = np.array(train_accs_a).mean()
                train_accs_b = np.array(train_accs_b).mean(axis=0)
                test_acc_a = np.array(test_accs_a).mean()
                test_accs_b = np.array(test_accs_b).mean(axis=0)
                self._logger.add_value("train_acc_a", train_acc_a)
                self._logger.add_value(
                        "train_acc_b/update_", train_accs_b.tolist())
                self._logger.add_value("test_acc_a", acc_a)
                self._logger.add_value(
                        "test_acc_b/update_", accs_b.tolist())
            self._logger.dump_summary(i)
        self._logger.save_tf_variables(
                self._model.get_variables(), i, self._sess)
        self._logger.close()
