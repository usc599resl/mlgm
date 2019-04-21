import io
import math

from matplotlib import pyplot as plt
import numpy as np


def transform_img(fig_arr):
    fig_arr = ((fig_arr + 1.) * 127.5).astype(np.int32)
    dim1 = math.ceil(math.sqrt(fig_arr.shape[0]))
    dim2 = fig_arr.shape[0] // dim1
    fig_arr = fig_arr.reshape((dim1, dim2) + fig_arr.shape[1:])
    return fig_arr


def get_matplotlib_fig(fig_arr):
    fig_shape_len = len(fig_arr.shape)
    assert fig_shape_len == 4 or fig_shape_len == 5
    cmap_config = None if fig_shape_len == 5 else "gray"
    fig, axs = plt.subplots(fig_arr.shape[0], fig_arr.shape[1])
    if len(axs.shape) == 1:
        axs = axs.reshape(axs.shape + (1, ))
    for i, j in np.ndindex(fig_arr.shape[:2]):
        axs[i][j].imshow(fig_arr[i][j], cmap=cmap_config)
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
