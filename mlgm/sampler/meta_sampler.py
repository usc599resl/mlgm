import numpy as np
import tensorflow as tf


class MetaSampler:
    def __init__(self, batch_size, meta_batch_size, train_classes,
                 test_classes, num_classes_per_batch, train_inputs,
                 train_labels, test_inputs, test_labels):
        assert num_classes_per_batch <= len(test_classes)
        # Duplicate the batch size since we need to sample twice from the
        # same distribution
        self._batch_size = batch_size * 2
        self._meta_batch_size = meta_batch_size
        self._train_classes = train_classes
        self._test_classes = test_classes
        self._num_classes_per_batch = num_classes_per_batch
        self._distribution = None
        self._train_ids_per_label = {}
        self._test_ids_per_label = {}
        self._train_inputs = train_inputs
        self._train_labels = train_labels
        self._test_inputs = test_inputs
        self._test_labels = test_labels
        self._dataset_sym, self._num_tasks = self._gen_train_metadata()
        self._input_batches, self._label_batches = self._gen_metabatch()

    @property
    def meta_batch_size(self):
        return self._meta_batch_size

    def _gen_train_metadata(self):
        raise NotImplementedError

    def _gen_metabatch(self):
        num_inputs_per_batch = self._batch_size * self._num_classes_per_batch
        num_inputs_per_meta_batch = (
            num_inputs_per_batch * self._meta_batch_size)

        meta_batch_dataset = self._dataset_sym.batch(num_inputs_per_meta_batch)
        meta_batch_itr = meta_batch_dataset.make_one_shot_iterator()
        meta_batch_sym = meta_batch_itr.get_next()

        all_input_batches = []
        all_label_batches = []
        for i in range(self._meta_batch_size):
            batch_input_sym = meta_batch_sym[0][i * num_inputs_per_batch:
                                                (i + 1) * num_inputs_per_batch]
            batch_label_sym = meta_batch_sym[1][i * num_inputs_per_batch:
                                                (i + 1) * num_inputs_per_batch]
            shuffle_batch_input_sym = []
            shuffle_batch_label_sym = []
            for k in range(self._batch_size):
                class_ids = tf.range(0, self._num_classes_per_batch)
                class_ids = tf.random_shuffle(class_ids)
                interleaved_class_ids = class_ids * self._batch_size + k
                train_instance_input_shuffle = tf.gather(
                    batch_input_sym, interleaved_class_ids)
                train_instance_label_shuffle = tf.gather(
                    batch_label_sym, interleaved_class_ids)
                shuffle_batch_input_sym.append(train_instance_input_shuffle)
                shuffle_batch_label_sym.append(train_instance_label_shuffle)

            shuffle_batch_input_sym = tf.concat(
                shuffle_batch_input_sym, axis=0)
            shuffle_batch_label_sym = tf.concat(
                shuffle_batch_label_sym, axis=0)
            all_input_batches.append(shuffle_batch_input_sym)
            all_label_batches.append(shuffle_batch_label_sym)
            
        all_input_batches = tf.stack(all_input_batches)
        all_label_batches = tf.stack(all_label_batches)
        return all_input_batches, all_label_batches

    def build_inputs_and_labels(self):
        raise NotImplementedError
