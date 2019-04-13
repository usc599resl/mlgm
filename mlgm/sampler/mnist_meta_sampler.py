from itertools import permutations
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from mlgm.sampler import MetaSampler


class MnistMetaSampler(MetaSampler):
    def __init__(
            self,
            batch_size,
            meta_batch_size,
            digits,
            num_classes_per_batch,
            one_hot_labels=False,
            same_input_and_label=False,
    ):
        assert digits is None or (
            type(digits) == list
            and [0 <= digit <= 9 for digit in digits])
        
        self._digits = list(set(digits))
        self._one_hot_labels = one_hot_labels
        self._same_input_and_label = same_input_and_label
        (train_inputs, train_labels), (test_inputs,
                                       test_labels) = mnist.load_data()

        inputs = np.concatenate((train_inputs, test_inputs))
        labels = np.concatenate((train_labels, test_labels))

        self._inputs_per_label = {}
        for digit in self._digits:
            ids = np.where(digit == labels)[0]
            random.shuffle(ids)
            self._inputs_per_label.update({digit: ids})

        inputs = inputs / 255.0
        super().__init__(batch_size, meta_batch_size, inputs, num_classes_per_batch)

    def _gen_metadata(self):
        all_ids = np.array([], dtype=np.int32)
        all_labels = np.array([], dtype=np.int32)
        num_tasks = 0
        for task in permutations(self._digits,
                                 self._num_classes_per_batch):
            task_ids = np.array([], dtype=np.int32)
            task_labels = np.array([], dtype=np.int32)
            for i, label in enumerate(task):
                label_ids = np.random.choice(
                    self._inputs_per_label[label], self._batch_size)
                labels = np.empty(self._batch_size, dtype=np.int32)
                labels.fill(i)
                task_labels = np.append(task_labels, labels)
                task_ids = np.append(task_ids, label_ids)
            all_labels = np.append(all_labels, task_labels)
            all_ids = np.append(all_ids, task_ids)
            num_tasks += 1
        all_ids_sym = tf.convert_to_tensor(all_ids)
        inputs_sym = tf.convert_to_tensor(self._inputs, dtype=tf.float32)
        all_inputs = tf.gather(inputs_sym, all_ids_sym)
        all_labels = tf.convert_to_tensor(
            all_labels, dtype=tf.dtypes.int32)
        if self._one_hot_labels:
            all_labels = tf.one_hot(all_labels, depth=10)
        dataset_sym = tf.data.Dataset.from_tensor_slices((all_inputs, all_labels))
        return dataset_sym, num_tasks

    def build_inputs_and_labels(self):
        slice_size = (self._batch_size // 2) * self._num_classes_per_batch
        input_a = tf.slice(self._input_batches, [0, 0, 0, 0],
                           [-1, slice_size, -1, -1])
        input_b = tf.slice(self._input_batches, [0, slice_size, 0, 0],
                           [-1, -1, -1, -1])
        if self._same_input_and_label:
            label_a = tf.reshape(input_a, input_a.get_shape().concatenate(1))
            label_b = tf.reshape(input_b, input_b.get_shape().concatenate(1))
        else:
            label_a = tf.slice(self._label_batches, [0, 0, 0],
                               [-1, slice_size, -1])
            label_b = tf.slice(self._label_batches, [0, slice_size, 0],
                               [-1, -1, -1])
        return input_a, label_a, input_b, label_b
