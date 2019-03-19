from gan import GAN
from dataset import Mnist
import tensorflow as tf

if __name__ == "__main__":
    mnist = Mnist('mnist')

    with tf.Session() as session:
        gan = GAN(mnist)
        gan.train(session)