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
        self._training_digits = list(set(train_digits))
        self._test_digits = list(set(test_digits))
        (train_inputs, train_labels), (test_inputs,
                                       test_labels) = mnist.load_data()
        train_inputs = train_inputs / 255.0
        test_inputs = test_inputs / 255.0
        super().__init__(batch_size, meta_batch_size, train_digits,
                         test_digits, num_classes_per_batch, train_inputs,
                         train_labels, test_inputs, test_labels)
