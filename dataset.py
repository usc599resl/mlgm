import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class Mnist:

    def __init__(self, name):
        self._name = name
        self.image_size = 28
        self.train_data = None
        self.num_trains = 0
        self._path = os.path.join('./data', self._name)
        self._load_mnist()

    def _load_mnist(self):
        self.train_data = input_data.read_data_sets(self._path, one_hot=False)
        self.num_trains = self.train_data.train.num_examples

    def _next_batch(self, batch_size):
        batch, _ = self.train_data.train.next_batch(batch_size)
        batch = batch.reshape([batch_size, self.image_size, self.image_size, 1])
        return batch