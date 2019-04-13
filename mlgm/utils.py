
import tensorflow as tf
import matplotlib.pyplot as plt
import io

def gen_fig(sess, input_imgs, gen_imgs):
    fig = plt.figure()
    for i, (input_img, gen_img) in enumerate(zip(input_imgs, gen_imgs)):
        nrows = 2
        ncolumns = input_imgs.shape[0]
        plt.subplot(nrows, ncolumns, (i + 1))
        plt.imshow(input_img[0], cmap='gray')
        plt.subplot(nrows, ncolumns, ncolumns + (i + 1))
        plt.imshow(gen_img[0].reshape(28, 28), cmap='gray')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image("plot", image)
    return sess.run(summary_op)