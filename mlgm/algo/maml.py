"""Simple MAML implementation.

Based on algorithm 1 from:
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning
for fast adaptation of deep networks." Proceedings of the 34th International
Conference on Machine Learning-Volume 70. JMLR. org, 2017.

https://arxiv.org/pdf/1703.03400.pdf
"""
import numpy as np
import tensorflow as tf


class Maml:
    def __init__(self,
                 model,
                 tasks,
                 num_updates=1,
                 alpha=1.0,
                 beta=1.0):
        self._model = model
        self._beta = beta
        self._num_updates = num_updates
        self._alpha = alpha
        self._tasks = tasks
        self._build()

    def _build(self):
        graph = tf.Graph()
        losses_b = []
        with graph.as_default():
            # Inputs
            input_a = self._tasks[0].build_input_placeholder(name="input_a")
            label_a = self._tasks[0].build_label_placeholder(name="label_a",
                    dtype=tf.dtypes.int32)
            input_b = self._tasks[0].build_input_placeholder(name="input_b")
            label_b = self._tasks[0].build_label_placeholder(name="label_b",
                    dtype=tf.dtypes.int32)
            alpha = tf.constant(self._alpha, name="alpha", dtype=input_a.dtype)
            # We build the model here to instantiate the weights of the layers
            self._model.build_forward_pass(input_a)


            for i in range(self._num_updates):
                losses_b.append(self._build_update(input_a, label_a, input_b,
                    label_b, alpha))
        writer = tf.summary.FileWriter(logdir='data/graphs', graph=graph)
        writer.flush()

    def _build_update(self, input_a, label_a, input_b, label_b, alpha):
        values=[input_a, label_a, input_b, label_b, alpha]
        loss_b = None
        with tf.variable_scope("update", values=values):
            output_a = self._model.build_forward_pass(input_a)
            loss_a = self._model.build_loss(label_a, output_a)
            grads, weights = self._model.build_compute_gradients(loss_a)
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                fast_weights = {
                        w.name: w - alpha * g for w, g in zip(weights, grads)
                }
            self._model.assign_model_params(fast_weights)
            output_b = self._model.build_forward_pass(input_b)
            loss_b = self._model.build_loss(label_b, output_b)
        return loss_b

    def train(self, sess, n_itr, restore_model_path=None):
        sess.run(tf.global_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        done = False
        for _ in range(n_itr):
            weights = None
            weights_prime = []
            losses_b = []
            for t_i in self._tasks:
                input_a, label_a, _ = t_i.sample()
                input_b, label_b, _ = t_i.sample()
                # This is self.total_loss1 in L81 of
                # https://github.com/cbfinn/maml/blob/master/maml.py
                loss_a = self._model.compute_loss(input_a, label_a)
                import pdb
                pdb.set_trace()
                weights, grads = self._model.compute_params_and_grads(loss_a)
                fast_weights.append({
                    x: weights[x] - self._alpha * grads[x]
                    for x in weights if x in grads
                })
                self._model.assign_model_params(fast_weights)
                losses_b.append(self._model.compute_loss(input_b, label_b))

                for i in range(num_updates - 1):
                    loss_a = self._model.compute_loss(input_a, label_a)
                    weights, grads = self._model.compute_params_and_grads(
                            loss_a)
                    fast_weights.append({
                        x: weights[x] - self._alpha * grads[x]
                        for x in weights if x in grads
                    })
                    self._model.assign_model_params(fast_weights)
                    losses_b.append(self._model.compute_loss(input_b, label_b))

                total_losses2 = [tf.reduce_mean(loss_b) for loss_b in losses_b]
                weights, grads = self.compute_params_and_grads(total_losses2[
                    num_updates - 1])
                # Clip gradients
                np.clip()

            sum_grads = None
            for t_i, theta_i in zip(self._tasks, theta_prime):
                self._model.assign_model_params(theta_i)
                t_i_x, t_i_y, _ = t_i.sample()
                _, grads = self._model.compute_params_and_grads(
                    t_i_x, t_i_y)
                if sum_grads is None:
                    sum_grads = grads
                else:
                    sum_grads = {
                        x: sum_grads[x] + grads[x]
                        for x in theta if x in grads
                    }

            theta = {
                x: theta[x] - self._beta * sum_grads[x]
                for x in theta if x in sum_grads
            }
            self._model.assign_model_params(theta)

            for i, t_i in enumerate(self._tasks):
                t_i_x, t_i_y = t_i.get_test_set()
                acc = self._model.compute_acc(t_i_x, t_i_y)
                print("Accuracy on test set {}: {}".format(i, acc))
