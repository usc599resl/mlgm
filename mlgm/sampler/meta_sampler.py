import numpy as np


class MetaSampler:
    def __init__(self, batch_size, meta_batch_size, train_classes,
                 test_classes, num_classes_per_batch, train_inputs,
                 train_labels, test_inputs, test_labels):
        assert num_classes_per_batch <= len(test_classes)
        self._batch_size = batch_size
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
        for train_class in train_classes:
            ids = np.where(train_class == self._train_labels)[0]
            self._train_ids_per_label.update({train_class: ids})
        for test_class in test_classes:
            ids = np.where(test_class == self._test_labels)[0]
            self._test_ids_per_label.update({test_class: ids})

    def sample_metabatch(self):
        self._distribution = np.array([
            np.random.choice(
                self._train_classes,
                size=self._num_classes_per_batch,
                replace=False) for _ in range(self._meta_batch_size)
        ])
        return self._sample_from_distribution(
            self._distribution, self._train_inputs, self._train_labels,
            self._train_ids_per_label)

    def sample_metabatch_from_previous_distribution(self):
        assert self._distribution is not None, "Call sample_metabatch() first"
        return self._sample_from_distribution(
            self._distribution, self._train_inputs, self._train_labels,
            self._train_ids_per_label)

    def _sample_from_distribution(self, distribution, inputs, labels,
                                  ids_per_label):
        meta_batch_inputs = None
        meta_batch_labels = None
        for sample in distribution:
            batch_ids = None
            for label in sample:
                ids = np.random.choice(
                    ids_per_label[label],
                    size=self._batch_size,
                    replace=False,
                )[np.newaxis].T
                if batch_ids is None:
                    batch_ids = ids
                else:
                    batch_ids = np.append(batch_ids, ids, axis=1)
            if meta_batch_inputs is None:
                meta_batch_inputs = inputs[batch_ids][np.newaxis]
                meta_batch_labels = labels[batch_ids][np.newaxis]
            else:
                meta_batch_inputs = np.append(
                    meta_batch_inputs, inputs[batch_ids][np.newaxis], axis=0)
                meta_batch_labels = np.append(
                    meta_batch_labels, labels[batch_ids][np.newaxis], axis=0)
        return meta_batch_inputs, meta_batch_labels

    def sample_test_batch(self):
        distribution = np.array([self._test_classes])
        return self._sample_from_distribution(distribution, self._test_inputs,
                                              self._test_labels,
                                              self._test_ids_per_label)
