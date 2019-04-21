import numpy as np
import tensorflow as tf
import ipdb

class MetaSampler:
    def __init__(self, batch_size, meta_batch_size, inputs, num_classes_per_batch):
        # Duplicate the batch size since we need to sample twice from the
        # same distribution
        self._batch_size = batch_size * 2
        self._meta_batch_size = meta_batch_size
        self._num_classes_per_batch = num_classes_per_batch
        self._distribution = None
        self._train_iterator = None
        self._test_iterator = None
        self._inputs = inputs
        self._ids_per_label = {}        
        self._train_dataset = self._gen_dataset()
        self._test_dataset = self._gen_dataset(test=True)

    @property
    def meta_batch_size(self):
        return self._meta_batch_size

    def restart_train_dataset(self, sess):
        sess.run(self._train_iterator.initializer)

    def restart_test_dataset(self, sess):
        sess.run(self._test_iterator.initializer)

    def _gen_dataset(self, test=False):
        raise NotImplementedError

    def init_iterators(self, sess):
        assert self._train_iterator
        assert self._test_iterator
        
        sess.run(self._train_iterator.initializer)
        sess.run(self._test_iterator.initializer)

        train_handle = sess.run(self._train_iterator.string_handle())
        test_handle = sess.run(self._test_iterator.string_handle())

        return train_handle, test_handle                             
            
    def _gen_metadata(self, handle):
        num_inputs_per_batch = (
            self._batch_size * self._num_classes_per_batch)
        self._train_iterator = \
                self._train_dataset.make_initializable_iterator()
        self._test_iterator = \
                self._test_dataset.make_initializable_iterator()

        iterator = tf.data.Iterator.from_string_handle(handle, 
            self._train_dataset.output_types, self._train_dataset.output_shapes)
        meta_batch_sym = iterator.get_next()
        all_input_batches = []
        all_label_batches = []
        for i in range(self._meta_batch_size):
            batch_input_sym = meta_batch_sym[0][i * num_inputs_per_batch:(
                i + 1) * num_inputs_per_batch]
            batch_label_sym = meta_batch_sym[1][i * num_inputs_per_batch:(
                i + 1) * num_inputs_per_batch]
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
                shuffle_batch_input_sym.append(
                    train_instance_input_shuffle)
                shuffle_batch_label_sym.append(
                    train_instance_label_shuffle)
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
