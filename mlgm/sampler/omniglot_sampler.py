from matplotlib import pyplot as plt
import numpy as np
import os

from mlgm.sampler import Sampler
from skimage.transform import resize


class OmniglotSampler(Sampler):
    def __init__(self,
                 batch_size=None,
                 one_hot_labels=False,
                 training_fraction=0.7):
        """Set the parameters to sample from Omniglot.

        - batch_size: number of training instances to use when sampling from
          Omniglot. If equal to None, the entire training set is used.
        """
        assert training_fraction < 1.

        x_train = 1. - np.load('dataset/Omniglot/Omniglot_training_data.npy').astype(np.float32)
        y_train = np.load('dataset/Omniglot/Omniglot_training_label.npy')
        x_test = 1. - np.load('dataset/Omniglot/Omniglot_testing_data.npy').astype(np.float32)
        y_test = np.load('dataset/Omniglot/Omniglot_testing_label.npy')

        x_train = resize(x_train, (x_train.shape[0], 28, 28), anti_aliasing=False)
        x_test = resize(x_test, (x_test.shape[0], 28, 28), anti_aliasing=False)

        num_of_class = 5

        ########
        # only get the first k classes
        if num_of_class != 50:
            idx_train_split = y_train.tolist().index([num_of_class])
            x_train = x_train[:idx_train_split, :, :]
            y_train = y_train[:idx_train_split, :]
            idx_test_split = y_test.tolist().index([num_of_class])
            x_test = x_test[:idx_test_split, :, :]
            y_test = y_test[:idx_test_split, :]
        ########

        if one_hot_labels:
            y_train_oh = np.zeros((y_train.shape[0], num_of_class))
            y_train = np.squeeze(y_train)
            y_train_oh[np.arange(y_train.size), y_train] = 1
            y_train = y_train_oh
            y_test_oh = np.zeros((y_test.shape[0], num_of_class))
            y_test = np.squeeze(y_test)
            y_test_oh[np.arange(y_test.size), y_test] = 1
            y_test = y_test_oh

        super().__init__(x_train, y_train, x_test, y_test, batch_size)

def num2str(idx):
    if idx < 10:
        return '0'+str(idx)
    return str(idx)

def load_img(fn):
    I = plt.imread(fn)
    I = np.array(I,dtype=np.uint8)
    return I


#######################################################################################
##################### Script for loading raw images into numpy array ##################

        # x_train = None
        # y_train = None
        # x_test = None
        # y_test = None

        # current_id = 0

        # img_dir = 'dataset/images_background'
        # alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.'] # get folder names

        # for alphabet in alphabet_names:
        #     num_of_samples_per_char = len(os.listdir(os.path.join(img_dir, alphabet)))
        #     x_temp = None
        #     y_temp = []
        #     print(num_of_samples_per_char)
        #     for j in range(1, num_of_samples_per_char + 1):
        #         img_char_dir = os.path.join(img_dir, alphabet, 'character' + num2str(j))
        #         all_imgs = os.listdir(img_char_dir)
        #         for file_name in all_imgs:
        #             fn_img = img_char_dir + '/' + file_name
        #             print(fn_img)
        #             img = load_img(fn_img)
        #             if x_temp is None:
        #                 x_temp = img[np.newaxis, :]
        #             else:
        #                 x_temp = np.concatenate((x_temp, img[np.newaxis, :]), axis=0)
        #             y_temp.append(current_id)
        #     y_temp = np.array(y_temp)[:, np.newaxis]

        #     # split
        #     n_total_sample = len(x_temp)
        #     n_training_sample = int(training_fraction * n_total_sample)

        #     idx = np.arange(n_total_sample)
        #     np.random.shuffle(idx)
        #     x_training_data = x_temp[idx[:n_training_sample]]
        #     x_testing_data = x_temp[idx[n_training_sample:]]
        #     y_training_data = y_temp[idx[:n_training_sample]]
        #     y_testing_data = y_temp[idx[n_training_sample:]]

        #     if current_id == 0:
        #         x_train = x_training_data
        #         y_train = y_training_data
        #         y_test = y_testing_data
        #         x_test = x_testing_data
        #     else:
        #         x_train = np.concatenate((x_train, x_training_data), axis=0)
        #         x_test = np.concatenate((x_test, x_testing_data), axis=0)
        #         y_train = np.concatenate((y_train, y_training_data), axis=0)
        #         y_test = np.concatenate((y_test, y_testing_data), axis=0)
        #     current_id += 1

        # img_dir = 'dataset/images_evaluation'
        # alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.'] # get folder names

        # for alphabet in alphabet_names:
        #     num_of_samples_per_char = len(os.listdir(os.path.join(img_dir, alphabet)))
        #     x_temp = None
        #     y_temp = []
        #     for j in range(1, num_of_samples_per_char + 1):
        #         img_char_dir = os.path.join(img_dir, alphabet, 'character' + num2str(j))
        #         all_imgs = os.listdir(img_char_dir)
        #         for file_name in all_imgs:
        #             fn_img = img_char_dir + '/' + file_name
        #             print(fn_img)
        #             img = load_img(fn_img)
        #             if x_temp is None:
        #                 x_temp = img[np.newaxis, :]
        #             else:
        #                 x_temp = np.concatenate((x_temp, img[np.newaxis, :]), axis=0)
        #             y_temp.append(current_id)
        #     y_temp = np.array(y_temp)[:, np.newaxis]

        #     # split
        #     n_total_sample = len(x_temp)
        #     n_training_sample = int(training_fraction * n_total_sample)

        #     idx = np.arange(n_total_sample)
        #     np.random.shuffle(idx)
        #     x_training_data = x_temp[idx[:n_training_sample]]
        #     x_testing_data = x_temp[idx[n_training_sample:]]
        #     y_training_data = y_temp[idx[:n_training_sample]]
        #     y_testing_data = y_temp[idx[n_training_sample:]]

        #     x_train = np.concatenate((x_train, x_training_data), axis=0)
        #     x_test = np.concatenate((x_test, x_testing_data), axis=0)
        #     y_train = np.concatenate((y_train, y_training_data), axis=0)
        #     y_test = np.concatenate((y_test, y_testing_data), axis=0)

        #     current_id += 1

        # np.save('Omniglot_training_data', x_train)
        # np.save('Omniglot_training_label', y_train)
        # np.save('Omniglot_testing_data', x_test)
        # np.save('Omniglot_testing_label', y_test)

        # import ipdb
        # ipdb.set_trace()