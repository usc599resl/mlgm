from itertools import permutations
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from mlgm.sampler import MetaSampler


class MnistMetaSampler(MetaSampler):
    def __init__(self, batch_size, meta_batch_size, train_digits, test_digits,
                 num_classes_per_batch):
        assert train_digits is None or (
            type(train_digits) == list
            and [0 <= train_digit <= 9 for train_digit in train_digits])
        assert test_digits is None or (
            type(test_digits) == list
            and [0 <= test_digit <= 9 for test_digit in test_digits])
        pass
        self._train_digits = list(set(train_digits))
        self._test_digits = list(set(test_digits))
        (train_inputs, train_labels), (test_inputs,
                                       test_labels) = mnist.load_data()

        self._train_inputs_per_label = {}
        self._test_inputs_per_label = {}
        for train_digit in self._train_digits:
            ids = np.where(train_digit == train_labels)[0]
            random.shuffle(ids)
            self._train_inputs_per_label.update({train_digit: ids})
        for test_digit in self._test_digits:
            ids = np.where(test_digit == test_labels)[0]
            random.shuffle(ids)
            self._test_inputs_per_label.update({test_digit: ids})

        train_inputs = train_inputs / 255.0
        test_inputs = test_inputs / 255.0
        super().__init__(batch_size, meta_batch_size, train_digits,
                         test_digits, num_classes_per_batch, train_inputs,
                         train_labels, test_inputs, test_labels)

    def _gen_train_metadata(self):
        all_train_ids = np.array([], dtype=np.int32)
        all_train_labels = np.array([], dtype=np.int32)
        num_tasks = 0
        for task in permutations(self._train_digits,
                                 self._num_classes_per_batch):
            task_ids = np.array([], dtype=np.int32)
            task_labels = np.array([], dtype=np.int32)
            for label in task:
                label_ids = np.random.choice(
                    self._train_inputs_per_label[label], self._batch_size)
                labels = np.empty(self._batch_size, dtype=np.int32)
                labels.fill(label)
                task_labels = np.append(task_labels, labels)
                task_ids = np.append(task_ids, label_ids)
            all_train_labels = np.append(all_train_labels, task_labels)
            all_train_ids = np.append(all_train_ids, task_ids)
            num_tasks += 1
        all_train_ids_sym = tf.convert_to_tensor(all_train_ids)
        train_inputs_sym = tf.convert_to_tensor(self._train_inputs)
        all_train_inputs = tf.gather(train_inputs_sym, all_train_ids_sym)
        all_train_labels = tf.convert_to_tensor(all_train_labels)
        dataset_sym = tf.data.Dataset.from_tensor_slices((all_train_inputs,
                                                          all_train_labels))
        return dataset_sym, num_tasks

    def build_inputs_and_labels(self):
        slice_size = (self._batch_size // 2) * self._num_classes_per_batch
        input_a = tf.slice(self._input_batches, [0, 0, 0, 0],
                           [-1, slice_size, -1, -1])
        input_b = tf.slice(self._input_batches, [0, slice_size, 0, 0],
                           [-1, -1, -1, -1])
        label_a = tf.slice(self._label_batches, [0, 0], [-1, slice_size])
        label_b = tf.slice(self._label_batches, [0, slice_size], [-1, -1])
        return input_a, label_a, input_b, label_b
