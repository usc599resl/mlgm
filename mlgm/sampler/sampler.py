import numpy as np


class Sampler:
    def __init__(self, x_train, y_train, x_test, y_test, batch_size=None):
        """Set the parameters to sample from a data set.

        - x_train: input features of the training set
        - y_train: labels of the training set
        - x_test: input features of the test set
        - y_test: labels of the test set
        - batch_size: number of training instances to use when sampling from
          the training_set. If equal to None, the entire training set is used.
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

        self._N = self._x_train.shape[0]
        assert batch_size is None or 0 <= batch_size <= self._N
        if batch_size:
            self._batch_size = batch_size
            self._sample_indices = np.array(range(0, self._N))
        else:
            self._batch_size = self._N

    def sample(self):
        if self._batch_size < self._N:
            end_of_epoch = False
            if self._sample_indices.size < self._batch_size:
                indices = self._sample_indices
                self._sample_indices = np.array(range(0, self._N))
                end_of_epoch = True
            else:
                indices = np.random.randint(0, len(self._sample_indices),
                                            self._batch_size)
                self._sample_indices = np.delete(self._sample_indices, indices)
            return self._x_train[indices], self._y_train[indices], end_of_epoch
        else:
            return self._x_train, self._y_train, True

    def get_test_set(self):
        return self._x_test, self._y_test
