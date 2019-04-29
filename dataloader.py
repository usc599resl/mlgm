import tensorflow as tf
import numpy as np
import scipy.io


class DataLoader:
	def __init__(self, dataset, training_digits=list(range(7)), testing_digits=[7,8,9]):
		dataset = dataset.lower()
		assert dataset in ['mnist', 'svhn']
		self.dataset_in_rgb = False
		if dataset == 'mnist':
			self._training_digits = training_digits
			self._testing_digits = testing_digits
			mnist = tf.keras.datasets.mnist
			(x_train, y_train),(x_test, y_test) = mnist.load_data()
			for i in range(10):
				if i not in self._training_digits:
					remove_train_indices = np.where(y_train == i)[0]
					x_train = np.delete(x_train, remove_train_indices, axis=0)
					y_train = np.delete(y_train, remove_train_indices, axis=0)
				if i not in self._testing_digits:
					remove_test_indices = np.where(y_test == i)[0]
					x_test = np.delete(x_test, remove_test_indices, axis=0)
					y_test = np.delete(y_test, remove_test_indices, axis=0)
			self.x_train = self.preprocess(x_train)
			self.y_train = y_train
			self.x_test = self.preprocess(x_test)
			self.y_test = y_test
		elif dataset == "svhn":
			self.dataset_in_rgb = True
			data = scipy.io.loadmat('SVHN_data/train_32x32.mat')
			x_train = data['X'].transpose(3,0,1,2)
			self.x_train = self.preprocess(x_train)
			self.y_train = data['y']
			data = scipy.io.loadmat('SVHN_data/test_32x32.mat')
			x_test = data['X'].transpose(3,0,1,2)
			self.x_test = self.preprocess(x_test)
			self.y_test = data['y']
		self.num_train_sample = self.x_train.shape[0]
		self.num_test_sample = self.x_test.shape[0]
		self.input_dim = np.prod(self.x_train[0].shape)
		self.sample_shape = self.x_train[0].shape

	def preprocess(self, data):
		return data.astype(np.float32) / 255.0

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
			return self.x_test[indices].reshape(sample_size, -1), self.y_test[indices].squeeze()
		else:
			indices = np.random.choice(self.num_train_sample, sample_size)
			return self.x_train[indices].reshape(sample_size, -1), self.y_train[indices].squeeze()

	def get_fixed_sample(self, indices):
		return self.x_test[indices].reshape(sample_size, -1)