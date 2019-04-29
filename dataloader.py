import tensorflow as tf
import numpy as np
import scipy.io
from skimage.transform import resize

class DataLoader:
	def __init__(self,
				 dataset,
				 training_digits=list(range(7)),
				 testing_digits=[7,8,9],
				 num_of_class=10):
		dataset = dataset.lower()
		assert dataset in ['mnist', 'fashionmnist', 'omniglot']
		if dataset == 'mnist':
			self._training_digits = training_digits
			self._testing_digits = testing_digits
			mnist = tf.keras.datasets.mnist
			(x_train, y_train),(x_test, y_test) = mnist.load_data()
			self.x_train_original_input = self.preprocess(x_train)
			for i in range(10):
				if i not in self._training_digits:
					remove_train_indices = np.where(y_train == i)[0]
					_x_train = np.delete(x_train, remove_train_indices, axis=0)
					_y_train = np.delete(y_train, remove_train_indices, axis=0)
				if i not in self._testing_digits:
					remove_test_indices = np.where(y_test == i)[0]
					_x_test = np.delete(x_test, remove_test_indices, axis=0)
					_y_test = np.delete(y_test, remove_test_indices, axis=0)
			self.x_train = self.preprocess(_x_train)
			self.y_train = _y_train
			self.x_test = self.preprocess(_x_test)
			self.y_test = _y_test
		elif dataset == 'fashionmnist':
			self._training_digits = training_digits
			self._testing_digits = testing_digits
			fashion_mnist = tf.keras.datasets.fashion_mnist
			(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
			self.x_train_original_input = self.preprocess(x_train)
			for i in range(10):
				if i not in self._training_digits:
					remove_train_indices = np.where(y_train == i)[0]
					_x_train = np.delete(x_train, remove_train_indices, axis=0)
					_y_train = np.delete(y_train, remove_train_indices, axis=0)
				if i not in self._testing_digits:
					remove_test_indices = np.where(y_test == i)[0]
					_x_test = np.delete(x_test, remove_test_indices, axis=0)
					_y_test = np.delete(y_test, remove_test_indices, axis=0)
			self.x_train = self.preprocess(_x_train)
			self.y_train = _y_train
			self.x_test = self.preprocess(_x_test)
			self.y_test = _y_test
		elif dataset == 'omniglot':
			x_train = 1. - np.load('dataset/Omniglot/Omniglot_training_data.npy').astype(np.float32)
			y_train = np.load('dataset/Omniglot/Omniglot_training_label.npy')
			x_test = 1. - np.load('dataset/Omniglot/Omniglot_testing_data.npy').astype(np.float32)
			y_test = np.load('dataset/Omniglot/Omniglot_testing_label.npy')

			x_train = resize(x_train, (x_train.shape[0], 28, 28), anti_aliasing=False)
			x_test = resize(x_test, (x_test.shape[0], 28, 28), anti_aliasing=False)

			########
			# only get the first k classes
			if num_of_class != 50:
				idx_train_split = y_train.tolist().index([num_of_class])
				_x_train = x_train[:idx_train_split, :, :]
				_y_train = y_train[:idx_train_split, :]
				idx_test_split = y_test.tolist().index([num_of_class])
				_x_test = x_test[:idx_test_split, :, :]
				_y_test = y_test[:idx_test_split, :]
			self.x_train = _x_train
			self.y_train = _y_train
			self.x_test = _x_test
			self.y_test = _y_test
			self.x_train_original_input = np.concatenate((_x_train, _x_test))
		# General
		self.num_train_sample = self.x_train.shape[0]
		self.num_test_sample = self.x_test.shape[0]
		self.input_dim = np.prod(self.x_train[0].shape)
		self.sample_shape = self.x_train[0].shape
		self.dataset = dataset

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

	def get_fixed_sample(self, sample_size, indices):
		return self.x_train_original_input[indices].reshape(sample_size, -1)
