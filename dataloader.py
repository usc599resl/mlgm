import tensorflow as tf
import numpy as np


class DataLoader:
	def __init__(self, dataset):
		assert dataset in ['mnist']

		if dataset == 'mnist':
			mnist = tf.keras.datasets.mnist
			(x_train, y_train),(x_test, y_test) = mnist.load_data()
			self.x_train = self.preprocess(x_train)
			self.y_train = y_train
			self.x_test = self.preprocess(x_test)
			self.y_test = y_test

			self.num_train_sample = self.x_train.shape[0]
			self.num_test_sample = self.x_test.shape[0]
			self.input_dim = np.prod(self.x_train[0].shape)

	def preprocess(self, data):
		return data / 255.0

	def sample(self, sample_size, test=False):
		if test:
			indices = np.random.choice(self.num_test_sample, sample_size)
			return self.x_test[indices].reshape(sample_size, -1)
		else:
			indices = np.random.choice(self.num_train_sample, sample_size)
			return self.x_train[indices].reshape(sample_size, -1)

	def sample_with_index(self, sample_size, test=False):
		if test:
			indices = np.random.choice(self.num_test_sample, sample_size)
			return self.x_test[indices].reshape(sample_size, -1), self.y_test[indices]
		else:
			indices = np.random.choice(self.num_train_sample, sample_size)
			return self.x_train[indices].reshape(sample_size, -1), self.y_train[indices]