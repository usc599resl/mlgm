import argparse
import tensorflow as tf
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from vae import VAE
from dataloader import DataLoader


def test_reconstruction(model, data_loader, batch_size=100):
    batch = data_loader.sample(batch_size, test=True)
    x_reconstructed = model.reconstruct(batch)

    n = np.sqrt(batch_size).astype(np.int32)
    w = data_loader.sample_shape[0]
    h = data_loader.sample_shape[1]
    if data_loader.dataset_in_rgb:
        c = data_loader.sample_shape[2]
        I_reconstructed = np.empty((h*n, 2*w*n, c))
    else:
        I_reconstructed = np.empty((h*n, 2*w*n))
    for i in range(n):
        for j in range(n):
            x = np.concatenate(
                (x_reconstructed[i*n+j, :].reshape(data_loader.sample_shape), 
                 batch[i*n+j, :].reshape(data_loader.sample_shape)),
                axis=1
            )
            I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

    plt.figure(figsize=(10, 20))
    plt.title('Odd column is generated image. Column number starts with 1')
    plt.imshow(I_reconstructed, cmap='gray')
    plt.show()

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
	data_loader = DataLoader(dataset='mnist') # or 'mnist'
	model = VAE(input_dim=data_loader.input_dim, latent_dim=16)

	log_step = 10
	batch_size = 32
	num_epoch = 100
	epoch_cycle = 1000

	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--load", help="load model or not", action="store_true")
	args = parser.parse_args()

	load_model = args.load

	with tf.Session() as sess:
		with tf.device('cpu:0'):
			saver = tf.train.Saver()
		# uncomment to use GPU
		# with tf.device('gpu:0'):
			sess.run(tf.global_variables_initializer())

			if load_model:
				path = './model/vae_model.ckpt'
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
				save_path = saver.save(sess, "./model/vae_model.ckpt")
				print("Model saved in path: %s" % save_path)

			################# Function below are for visualization ####################
			test_reconstruction(model, data_loader)
			# test_transformation(model, data_loader)
			# test_interpolation(model, data_loader)
			###########################################################################
	print('Done!')


