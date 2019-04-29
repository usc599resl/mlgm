import argparse
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

from vae import VAE
from dataloader import DataLoader


def test_reconstruction(model, data_loader, batch_size=7):
    # batch = data_loader.sample(batch_size, test=True)
    # fixed testing images
    ind = {
    	'mnist': [1404, 806, 798, 1519, 930, 884, 1666],
    	'fashionmnist': [1234, 867, 770, 1360, 962, 848, 1475],
    	'omniglot': [3649, 4168, 5199, 4174, 3652, 3655, 4170]}
    indices = ind[data_loader.dataset]
    batch = data_loader.get_fixed_sample(batch_size, indices)
    x_reconstructed = model.reconstruct(batch)

    w = data_loader.sample_shape[0]
    h = data_loader.sample_shape[1]
    I_reconstructed = np.empty((h*2, w*batch_size))
    for i in range(batch_size):
        I_reconstructed[:h, i*w:(i+1)*w] = batch[i, :].reshape(data_loader.sample_shape)
        I_reconstructed[h:, i*w:(i+1)*w] = x_reconstructed[i, :].reshape(data_loader.sample_shape)

    plt.figure(figsize=(10, 20))
    plt.title('Odd column is generated image. Column number starts with 1')
    plt.imshow(I_reconstructed, cmap='gray')
    plt.axis('off')
    # plt.show()
    plt.savefig('result_vae.png')

def test_transformation(model, data_loader, batch_size=2000):
    batch, idx = data_loader.sample_with_index(batch_size)
    z = model.transform(batch)
    plt.figure(figsize=(10, 8)) 
    plt.scatter(z[:, 0], z[:, 1], c=idx, cmap='brg')
    plt.colorbar()
    plt.show()

def test_interpolation(model, data_loader, num=None):
	"""
	Currently only works when latent_dim=2
	"""
	assert model.latent_dim == 2, "Test interpolation only works for latent_dim=2 now"
	n = 30
	width = data_loader.sample_shape[0]
	height = data_loader.sample_shape[1]
	if data_loader.dataset_in_rgb:
		channel = data_loader.sample_shape[2]
		figure = np.zeros((width * n, height * n, channel))
	else:
		figure = np.zeros((width * n, height * n))

	if not num:
		grid_x = np.linspace(-2., 2., n)
		grid_y = np.linspace(-2., 2., n)
	else:
		grid_x, grid_y = get_z_range(num, n)

	for i, yi in enumerate(grid_y):
	    for j, xi in enumerate(grid_x):
	        z_sample = np.array([[xi, yi]])
	        digit = model.reconstruct_from_z(z_sample).reshape(data_loader.sample_shape)
	        figure[i * width: (i + 1) * width,
	               j * height: (j + 1) * height] = digit

	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='gray')
	plt.show()

def get_z_range(num, n):
	# only illustrate number 1 and 2 for MNIST
	assert num == 1 or num == 2, ("Only illustrate number 1 and 2. "
	"Find the latent range yourself for other numbers.")

	if num == 1:
		grid_x = np.linspace(-2, -1.3, n)
		grid_y = np.linspace(-1, 0.9, n)
	elif num == 2:
		grid_x = np.linspace(0.3, 0.4, n)
		grid_y = np.linspace(-0.5, 0., n)

	return grid_x, grid_y

if __name__=="__main__":
	dataset = 'fashionmnist' # mnist / fashionmnist / omniglot
	load_from = 'omniglot'
	data_loader = DataLoader(dataset=dataset)
	model = VAE(input_dim=data_loader.input_dim, latent_dim=32)

	log_step = 10
	batch_size = 32
	num_epoch = 500
	epoch_cycle = 100

	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--load", help="load model or not", action="store_true")
	args = parser.parse_args()

	load_model = args.load

	with tf.Session() as sess:
		with tf.device('cpu:0'):
			saver = tf.train.Saver()
		# uncomment to use GPU
		with tf.device('gpu:0'):
			sess.run(tf.global_variables_initializer())

			if load_model:
				path = './model/vae_model_{}.ckpt'.format(load_from)
				saver.restore(sess, path)
				print("Model loaded from path: %s" % path)
			else:
				for epoch in range(num_epoch):
					start_time = time.time()
					for _ in range(epoch_cycle):
						batch = data_loader.sample(batch_size)
						########################################################################
						## This part of code is only for testing if forward and loss_func works
						## loss_f_val is similar to losses 
						# x_hat, mu_out, log_sigma_out = model.forward(batch, model.get_weights())
						# loss_f = model.loss_func(x_hat, mu_out, log_sigma_out)
						# loss_f_val = sess.run(loss_f, feed_dict={model.x: batch})
						########################################################################
						losses = model.optimize(batch)
						end_time = time.time()

					if epoch % log_step == 0:
						log_str = '[Epoch {}] \n'.format(epoch)
						for k, v in losses.items():
							log_str += '{}: {:.3f}  '.format(k, v)
							log_str += '({:.3f} sec/epoch) \n'.format(end_time - start_time)
						print(log_str)
				save_path = saver.save(sess, "./model/vae_model_{}.ckpt".format(dataset))
				print("Model saved in path: %s" % save_path)

			################# Function below are for visualization ####################
			test_reconstruction(model, data_loader)
			# test_transformation(model, data_loader)
			# test_interpolation(model, data_loader)
			###########################################################################
	print('Done!')


