import pdb
import numpy as np
import pickle

with open("data/cifar-10-batches-py/data_batch_1", "rb") as fo:
    cifar_10_1 = pickle.load(fo, encoding="bytes")
    labels = np.array(cifar_10_1[b'labels'])

pdb.set_trace()
pass
