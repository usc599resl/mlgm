from itertools import permutations
import pickle

import numpy as np
import tensorflow as tf

from mlgm.sampler import MetaSampler


class Cifar10MetaSampler(MetaSampler):
    def __init__(
            self,
            batch_size,
            meta_batch_size,
            train_classes,
            test_classes,
            num_classes_per_batch,
            one_hot_labels=False,
            same_input_and_label=False,
    ):
        self._train_classes = list(set(train_classes))
        self._test_classes = list(set(test_classes))
        self._one_hot_labels = one_hot_labels
        self._same_input_and_label = same_input_and_label

        with open("data/cifar-10-batches-py/batches.meta", "rb") as f:
            cifar_10_meta = pickle.load(f, encoding="bytes")
            self._label_names = cifar_10_meta[b'label_names']

        self._train_inputs_per_label = {}
        self._test_inputs_per_label = {}
        with open("data/cifar-10-batches-py/data_batch_1", "rb") as f:
            train_batch = pickle.load(f, encoding="bytes")
            train_inputs = np.array(train_batch[b'data']).reshape(-1, 3, 32,
                                                              32).transpose(
                                                                  0, 2, 3, 1)
            train_labels = np.array(train_batch[b'labels'])
            for train_class in self._train_classes:
                ids = np.argwhere(train_class == train_labels).reshape(-1)
                np.random.shuffle(ids)
                self._train_inputs_per_label.update({train_class: ids})

        with open("data/cifar-10-batches-py/test_batch", "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")
            test_inputs = np.array(test_batch[b'data']).reshape(-1, 3, 32,
                                                                32).transpose(
                                                                    0, 2, 3, 1)
            test_labels = np.array(test_batch[b'labels'])
            for test_class in self._test_classes:
                ids = np.argwhere(test_class == test_labels).reshape(-1)
                np.random.shuffle(ids)
                self._test_inputs_per_label.update({test_class: ids})

        super().__init__(batch_size, meta_batch_size, train_classes,
                         test_classes, num_classes_per_batch, train_inputs,
                         train_labels, test_inputs, test_labels)

    def _gen_train_metadata(self):
        dataset_sym, num_tasks = self._gen_metadata(
            self._train_inputs_per_label, self._train_classes,
            self._train_inputs)
        return dataset_sym, num_tasks

    def _gen_test_metadata(self):
        dataset_sym, num_tasks = self._gen_metadata(
            self._test_inputs_per_label, self._test_classes, self._test_inputs)
        return dataset_sym, num_tasks

    def _gen_metadata(self, inputs_per_label, digits, inputs):
        all_ids = np.array([], dtype=np.int32)
        all_labels = np.array([], dtype=np.int32)
        num_tasks = 0
        for task in permutations(digits, self._num_classes_per_batch):
            task_ids = np.array([], dtype=np.int32)
            task_labels = np.array([], dtype=np.int32)
            for i, label in enumerate(task):
                label_ids = np.random.choice(inputs_per_label[label],
                                             self._batch_size)
                labels = np.empty(self._batch_size, dtype=np.int32)
                labels.fill(i)
                task_labels = np.append(task_labels, labels)
                task_ids = np.append(task_ids, label_ids)
            all_labels = np.append(all_labels, task_labels)
            all_ids = np.append(all_ids, task_ids)
            num_tasks += 1
        all_ids_sym = tf.convert_to_tensor(all_ids)
        inputs_sym = tf.convert_to_tensor(inputs, dtype=tf.int32)
        all_inputs_sym = tf.gather(inputs_sym, all_ids_sym)
        all_labels_sym = tf.convert_to_tensor(
            all_labels, dtype=tf.dtypes.int32)
        if self._one_hot_labels:
            all_labels_sym = tf.one_hot(all_labels_sym, depth=10)
        dataset_sym = tf.data.Dataset.from_tensor_slices((all_inputs_sym,
                                                          all_labels_sym))
        return dataset_sym, num_tasks

    def _build_inputs_and_labels(self, input_batches, label_batches):
        slice_size = (self._batch_size // 2) * self._num_classes_per_batch
        input_a = tf.slice(input_batches, [0, 0, 0, 0, 0],
                           [-1, slice_size, -1, -1, -1])
        input_b = tf.slice(input_batches, [0, slice_size, 0, 0, 0],
                           [-1, -1, -1, -1, -1])
        if self._same_input_and_label:
            label_a = input_a
            label_b = input_b
        else:
            if self._one_hot_labels:
                label_a = tf.slice(label_batches, [0, 0, 0],
                                   [-1, slice_size, -1])
                label_b = tf.slice(label_batches, [0, slice_size, 0],
                                   [-1, -1, -1])
            else:
                label_a = tf.slice(label_batches, [0, 0], [-1, slice_size])
                label_b = tf.slice(label_batches, [0, slice_size], [-1, -1])
        return input_a, label_a, input_b, label_b

    def build_train_inputs_and_labels(self):
        return self._build_inputs_and_labels(self._train_input_batches,
                                             self._train_label_batches)

    def build_test_inputs_and_labels(self):
        return self._build_inputs_and_labels(self._test_input_batches,
                                             self._test_label_batches)
