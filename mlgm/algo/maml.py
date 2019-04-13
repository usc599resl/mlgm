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
import matplotlib.pyplot as plt
import io

class Maml:
    def __init__(self,
                 model,
                 metasampler,
                 sess,
                 compute_acc=True,
                 name="maml",
                 num_updates=1,
                 update_lr=0.0001,
                 meta_lr=0.0001):
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._compute_acc = compute_acc
        self._num_updates = num_updates
        self._update_lr = update_lr
        self._meta_lr = meta_lr
        self._logger = Logger(name) 
        self._build()
        self._logger.add_graph(self._sess.graph)

    def _build(self):
        with self._sess.graph.as_default():
            (self._input_a, self._label_a, self._input_b,
             self._label_b) = self._metasampler.build_inputs_and_labels()

            def task_metalearn(args):
                input_a, label_a, input_b, label_b = args
                output = self._model.build_forward_pass(input_a)
                acc = self._model.build_accuracy(label_a, output)

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

                return output, loss_a, acc_a, losses_b, accs_b

            out_dtype = (tf.float32, tf.float32, tf.float32,
                         [tf.float32] * self._num_updates,
                         [tf.float32] * self._num_updates)
            elems = (self._input_a, self._label_a, self._input_b,
                     self._label_b)
            (self._gen_images, self._loss_a, self._acc_a, self._losses_b,
             self._accs_b) = tf.map_fn(
                 task_metalearn,
                 elems=elems,
                 dtype=out_dtype,
                 parallel_iterations=self._metasampler.meta_batch_size)

            with tf.variable_scope("metatrain", values=[self._losses_b]):
                self._metatrain_op = tf.train.AdamOptimizer(
                    self._meta_lr).minimize(
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
        return loss_a, acc_a, loss_b, acc_b, new_fast_weights

    def _compute_metatrain_and_acc(self):
        loss_a, acc_a, losses_b, accs_b, _ = self._sess.run([
            self._loss_a, self._acc_a, self._losses_b, self._accs_b,
            self._metatrain_op
        ])
        return loss_a, acc_a, losses_b, accs_b

    def _compute_metatest_and_acc(self):
        input_imgs, gen_imgs, loss_a, acc_a, losses_b, accs_b = self._sess.run([
            self._input_a, self._gen_images, self._loss_a, 
            self._acc_a, self._losses_b, self._accs_b
        ])
        return input_imgs, gen_imgs, loss_a, acc_a, losses_b, accs_b

    def _compute_metatrain(self):
        loss_a, losses_b, _ = self._sess.run(
            [self._loss_a, self._losses_b, self._metatrain_op])
        return loss_a, losses_b

    def _compute_metatest(self):
        input_imgs, gen_imgs, loss_a, losses_b = self._sess.run([
            self._input_a, self._gen_images, self._loss_a, self._losses_b
        ])
        return input_imgs, gen_imgs, loss_a, losses_b 

    def train(self, num_iterations, restore_model_path=None):
        return self._run(num_iterations, restore_model_path, False)

    def test(self, num_iterations, restore_model_path):
        return self._run(num_iterations, restore_model_path, True)

    def _run(self, num_iterations, restore_model_path, test):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)
        self._metasampler.restart_dataset(self._sess)
        for i in range(num_iterations):
            try:
                if self._compute_acc:
                    if test:
                        input_imgs, gen_imgs, loss_a, acc_a, losses_b, accs_b = self._compute_metatest_and_acc()            
                        self._logger.add_image(self.gen_fig(input_imgs, gen_imgs), i)            
                    else:
                        loss_a, acc_a, losses_b, accs_b = self._compute_metatrain_and_acc()
                    acc_a = np.mean(acc_a)
                    accs_b = np.array(accs_b).mean(axis=1)
                else:
                    if test:
                        input_imgs, gen_imgs, loss_a, losses_b = self._compute_metatest()   
                        self._logger.add_image(self.gen_fig(input_imgs, gen_imgs), i)                                 
                    else:
                        loss_a, losses_b = self._compute_metatrain()
                                        
                loss_a = np.mean(loss_a)
                losses_b = np.array(losses_b).mean(axis=1)
                self._logger.new_summary()
                self._logger.add_value("loss_a", loss_a)
                self._logger.add_value("loss_b/update_", losses_b.tolist())
                if self._compute_acc:
                    self._logger.add_value("acc_a", acc_a)
                    self._logger.add_value("acc_b/update_", accs_b.tolist())
                self._logger.dump_summary(i)
                self._logger.save_tf_variables(self._model.get_variables(), i, self._sess)
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_dataset(self._sess)
        self._logger.close()

    def gen_fig(self, imgs_a, gen_imgs_a):
        fig = plt.figure()
        for i, (img_a, gen_img_a) in enumerate(zip(imgs_a, gen_imgs_a)):
            plt.subplot(2, 3, (i + 1))
            plt.imshow(img_a[0], cmap='gray')
            plt.subplot(2, 3, 3 + (i + 1))
            plt.imshow(gen_img_a[0].reshape(28, 28), cmap='gray')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        summary_op = tf.summary.image("plot", image)
        return self._sess.run(summary_op)
