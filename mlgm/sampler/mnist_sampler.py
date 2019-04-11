import numpy as np
from tensorflow.keras.datasets import mnist

from mlgm.sampler import Sampler


class MnistSampler(Sampler):
    def __init__(self,
                 training_digits=None,
                 batch_size=None,
                 one_hot_labels=False):
        """Set the parameters to sample from MNIST.

        - training_digits: list of digits to use for training, e.g., if equal
          to [3, 4] only 3 and 4 digits are sampled for training.
        - batch_size: number of training instances to use when sampling from
          MNIST. If equal to None, the entire training set is used.
        """
        assert training_digits is None or (
            type(training_digits) == list
            and [0 <= train_dig <= 9 for train_dig in training_digits])

        if training_digits:
            # Remove any possible duplicates with set
            self._training_digits = list(set(training_digits))
        else:
            self._training_digits = list(range(10))

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        for i in range(10):
            if i not in self._training_digits:
                remove_train_indices = np.where(y_train == i)[0]
                x_train = np.delete(x_train, remove_train_indices, axis=0)
                y_train = np.delete(y_train, remove_train_indices, axis=0)
                remove_test_indices = np.where(y_test == i)[0]
                x_test = np.delete(x_test, remove_test_indices, axis=0)
                y_test = np.delete(y_test, remove_test_indices, axis=0)
        if one_hot_labels:
            y_train_oh = np.zeros(y_train.shape + (10, ))
            y_train_oh[np.arange(y_train.size), y_train] = 1
            y_train = y_train_oh
            y_test_oh = np.zeros(y_test.shape + (10, ))
            y_test_oh[np.arange(y_test.size), y_test] = 1
            y_test = y_test_oh
        super().__init__(x_train, y_train, x_test, y_test, batch_size)
