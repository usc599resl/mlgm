from itertools import permutations
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

from mlgm.sampler import MetaSampler

class FashionMnistMetaSampler(MetaSampler):
    def __init__(
            self,
            batch_size,
            meta_batch_size,
            train_digits,
            test_digits,
            num_classes_per_batch,
            one_hot_labels=False,
            same_input_and_label=False,
    ):
        assert train_digits is None or (
            type(train_digits) == list
            and [0 <= digit <= 9 for digit in train_digits])

        assert test_digits is None or (
            type(test_digits) == list
            and [0 <= digit <= 9 for digit in test_digits])
        
        self._train_digits = list(set(train_digits))
        self._test_digits = list(set(test_digits))
        self._one_hot_labels = one_hot_labels
        self._same_input_and_label = same_input_and_label
        (train_inputs, train_labels), (test_inputs,
                                       test_labels) = fashion_mnist.load_data()

        inputs = np.concatenate((train_inputs, test_inputs))
        labels = np.concatenate((train_labels, test_labels))

        self._train_inputs_per_label = {}
        self._test_inputs_per_label = {}
        self._train_size = 0
        self._test_size = 0

        for digit in self._train_digits:
            ids = np.where(digit == labels)[0]
            self._train_size += len(ids)
            random.shuffle(ids)
            self._train_inputs_per_label.update({digit: ids})

        for digit in self._test_digits:
            ids = np.where(digit == labels)[0]
            self._test_size += len(ids)
            random.shuffle(ids)
            self._test_inputs_per_label.update({digit: ids})

        inputs = inputs / 255.0
        super().__init__(batch_size, meta_batch_size, inputs, num_classes_per_batch)

    def _gen_dataset(self, test=False):
        digits = self._test_digits if test else self._train_digits
        inputs_per_label = self._test_inputs_per_label if test else self._train_inputs_per_label

        tasks = []
        while True:
            tasks_remaining = self._meta_batch_size - len(tasks)
            if tasks_remaining <= 0:
                break            
            tasks_to_add = list(permutations(digits, self._num_classes_per_batch))
            n_tasks_to_add = min(len(tasks_to_add), tasks_remaining)
            tasks.extend(tasks_to_add[:n_tasks_to_add])            

        num_inputs_per_meta_batch = (self._batch_size * 
            self._num_classes_per_batch * self._meta_batch_size)
        
        ids = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32)
        lbls = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32)

        data_size = self._test_size if test else self._train_size
        data_size = data_size // num_inputs_per_meta_batch
        data_size = min(data_size, 1000)
                        
        for i in range(data_size):
            all_ids = np.array([], dtype=np.int32)
            all_labels = np.array([], dtype=np.int32)
            for task in tasks:
                task_ids = np.array([], dtype=np.int32)
                task_labels = np.array([], dtype=np.int32)
                for i, label in enumerate(task):
                    label_ids = np.random.choice(inputs_per_label[label], self._batch_size)
                    labels = np.empty(self._batch_size, dtype=np.int32)
                    labels.fill(i)
                    task_labels = np.append(task_labels, labels)
                    task_ids = np.append(task_ids, label_ids)
                all_labels = np.append(all_labels, task_labels)
                all_ids = np.append(all_ids, task_ids)
            ids = np.append(ids, [all_ids], axis=0)
            lbls = np.append(lbls, [all_labels], axis=0)

        all_ids_sym = tf.convert_to_tensor(ids)
        inputs_sym = tf.convert_to_tensor(self._inputs, dtype=tf.float32)
        all_inputs = tf.gather(inputs_sym, all_ids_sym)
        all_labels = tf.convert_to_tensor(
            lbls, dtype=tf.dtypes.int32)
        if self._one_hot_labels:
            all_labels = tf.one_hot(all_labels, depth=10)
        dataset_sym = tf.data.Dataset.from_tensor_slices((all_inputs, all_labels))
        return dataset_sym

    def build_inputs_and_labels(self, handle):
        slice_size = (self._batch_size // 2) * self._num_classes_per_batch
        input_batches, label_batches = self._gen_metadata(handle)

        input_a = tf.slice(input_batches, [0, 0, 0, 0],
                           [-1, slice_size, -1, -1])
        input_b = tf.slice(input_batches, [0, slice_size, 0, 0],
                           [-1, -1, -1, -1])
        if self._same_input_and_label:
            label_a = tf.reshape(input_a, input_a.get_shape().concatenate(1))
            label_b = tf.reshape(input_b, input_b.get_shape().concatenate(1))
        else:
            label_a = tf.slice(label_batches, [0, 0, 0],
                               [-1, slice_size, -1])
            label_b = tf.slice(label_batches, [0, slice_size, 0],
                               [-1, -1, -1])
        return input_a, label_a, input_b, label_b
