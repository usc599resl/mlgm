from datetime import datetime
import os

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Model:
    def __init__(self,
                 layers,
                 param_in,
                 param_out,
                 sess,
                 learning_rate=0.001,
                 model_name="model"):
        assert len(layers) > 0 and [type(layer) is Layer for layer in layers]
        self._sess = sess
        self._x = None
        self._y = None
        self._out = None
        self._param_in = param_in
        self._param_out = param_out
        self._layers = layers
        self._params = None
        self._grads = None
        self._acc = None
        self._optimize = None
        self._learning_rate = learning_rate
        self._name = model_name
        self._build_model()
        self._build_gradients()
        self._build_accuracy()
        var_list = [
            var for var in tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        ]
        self._saver = tf.train.Saver(var_list)

    def _build_model(self):
        # Model input/output
        self._x = tf.placeholder(**self._param_in)
        self._y = tf.placeholder(**self._param_out)
        layer_in = self._x
        # Model layers
        with tf.variable_scope(self._name, values=[layer_in]):
            for layer in self._layers:
                layer_out = layer(layer_in)
                layer_in = layer_out
            self._out = layer_out

    def _build_gradients(self):
        loss = tf.losses.sparse_softmax_cross_entropy(self._y, self._out)
        adam_opt = tf.train.AdamOptimizer(self._learning_rate)
        grad_var = adam_opt.compute_gradients(loss)

        # Gradients where keys are the TF variable names
        self._grads = {}
        # Model parameters where keys are the TF variable names
        self._params = {}
        for grad, var in grad_var:
            self._grads.update({var.name: grad})
            self._params.update({var.name: var})
        self._optimize = adam_opt.apply_gradients(grad_var)

    def _build_accuracy(self):
        # Calculate accuracy
        y_pred = tf.math.argmax(self._out, axis=1)
        self._acc = tf.reduce_mean(
            tf.cast(tf.equal(y_pred, self._y), tf.float32))

    def compute_params_and_grads(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        return self._sess.run([self._params, self._grads], feed_dict=feed_dict)

    def optimize(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        self._sess.run([self._optimize], feed_dict=feed_dict)

    def compute_acc(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        return self._sess.run(self._acc, feed_dict=feed_dict)

    def assign_model_params(self, params):
        assign_ops = []
        for i in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name):
            if i.name in params:
                assign_ops.append(i.assign(params[i.name]))
        self._sess.run(assign_ops)

    def save_model(self):
        model_path = "data/" + self._name + "_" + datetime.now().strftime(
            "%H_%M_%m_%d_%y")
        os.makedirs(model_path)
        self._saver.save(self._sess, model_path + "/" + self._name)

    def restore_model(self, save_path):
        self._saver.restore(self._sess, save_path)
