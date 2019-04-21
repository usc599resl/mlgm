import pickle

import numpy as np
import tensorflow as tf


class Cifar10Sampler:
    def __init__(self, batch_size):
        with open("data/cifar-10-batches-py/batches.meta", "rb") as f:
            cifar_10_meta = pickle.load(f, encoding="bytes")
            self._label_names = cifar_10_meta[b'label_names']

        all_train_inputs = None
        all_train_labels = None
        for i in range(1, 6):
            with open("data/cifar-10-batches-py/data_batch_" + str(i),
                      "rb") as f:
                train_batch = pickle.load(f, encoding="bytes")
                train_inputs = np.array(train_batch[b'data']).reshape(
                    -1, 3, 32, 32).transpose(0, 2, 3, 1)
                train_inputs = (train_inputs / 127.5) - 1
                train_labels = np.array(train_batch[b'labels'])
                if all_train_inputs is None:
                    all_train_inputs = train_inputs
                else:
                    all_train_inputs = np.append(
                        all_train_inputs, train_inputs, axis=0)
                if all_train_labels is None:
                    all_train_labels = train_labels
                else:
                    all_train_labels = np.append(
                        all_train_labels, train_labels, axis=0)
        train_inputs = all_train_inputs
        train_labels = all_train_labels

        with open("data/cifar-10-batches-py/test_batch", "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")
            test_inputs = np.array(test_batch[b'data']).reshape(-1, 3, 32,
                                                                32).transpose(
                                                                    0, 2, 3, 1)
            test_inputs = (test_inputs / 127.5) - 1
            test_labels = np.array(test_batch[b'labels'])

        train_in_sym = tf.convert_to_tensor(train_inputs)
        train_lb_sym = tf.convert_to_tensor(train_labels)
        test_in_sym = tf.convert_to_tensor(test_inputs)
        test_lb_sym = tf.convert_to_tensor(test_labels)
        train_dataset_sym = tf.data.Dataset.from_tensor_slices((train_in_sym,
                                                                train_lb_sym))
        test_dataset_sym = tf.data.Dataset.from_tensor_slices((test_in_sym,
                                                               test_lb_sym))
        train_batch_dataset = train_dataset_sym.batch(
            batch_size, drop_remainder=True)
        train_batch_itr = train_batch_dataset.make_initializable_iterator()
        test_batch_dataset = test_dataset_sym.batch(
            batch_size, drop_remainder=True)
        test_batch_itr = test_batch_dataset.make_initializable_iterator()
        self._train_batch_itr = train_batch_itr
        self._test_batch_itr = test_batch_itr

    def restart_train_dataset(self, sess):
        assert self._train_batch_itr
        sess.run(self._train_batch_itr.initializer)

    def restart_test_dataset(self, sess):
        assert self._test_batch_itr
        sess.run(self._test_batch_itr.initializer)

    def build_train_inputs_and_labels(self):
        train_batch = self._train_batch_itr.get_next()
        return train_batch[0], train_batch[1]

    def build_test_inputs_and_labels(self):
        test_batch = self._test_batch_itr.get_next()
        return test_batch[0], test_batch[1]
