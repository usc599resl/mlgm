import io
import math

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def get_matplotlib_fig(fig_arr):
    if len(fig_arr.shape) == 4:
        fig, axs = plt.subplots(fig_arr.shape[0], fig_arr.shape[1])
        if len(axs.shape) == 1:
            axs = axs.reshape(axs.shape + (1, ))
        for i, j in np.ndindex(fig_arr.shape[:2]):
            axs[i][j].imshow(fig_arr[i][j], cmap='gray')
    return fig

def get_img_from_arr(fig_arr):
    fig = get_matplotlib_fig(fig_arr)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))
