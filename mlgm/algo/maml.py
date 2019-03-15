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
                 alpha=1.0,
                 beta=1.0):
        self._model = model
        self._beta = beta
        self._alpha = alpha
        self._tasks = tasks

    def train(self, sess, n_itr, restore_model_path=None):
        sess.run(tf.global_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        done = False
        for _ in range(n_itr):
            theta = None
            theta_prime = []
            for t_i in self._tasks:
                t_i_x, t_i_y, _ = t_i.sample()
                theta, grads = self._model.compute_params_and_grads(
                    t_i_x, t_i_y)
                theta_prime.append({
                    x: theta[x] - self._alpha * grads[x]
                    for x in theta if x in grads
                })

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
