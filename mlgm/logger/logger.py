from datetime import datetime
import os

import tensorflow as tf
from tensorflow import summary as summ


class Logger:
    def __init__(self, exp_name, graph=None):
        self._log_path = "data/" + exp_name + "_" + datetime.now().strftime(
            "%H_%M_%m_%d_%y")
        self._writer = summ.FileWriter(self._log_path, graph)
        self._summary_mrg = None
        self._writer.flush()

    def new_summary(self):
        self._summary = tf.Summary()
        self._std_out = {}

    def add_value(self, name, value):
        self._summary.value.add(tag=name, simple_value=value)
        self._std_out.update({name: value})

    def add_graph(self, graph):
        self._writer.add_graph(graph)

    @property
    def summary(self):
        return self._summary_mrg

    def dump_summary(self, itr):
        self._writer.add_summary(self._summary, itr)
        self._writer.flush()
        print("--------------------------------------------------")
        print("exp_name: {}".format(self._log_path))
        print("itr: {}".format(itr))
        for k, v in self._std_out.items():
            print("{}: {}".format(k, v))
        print("--------------------------------------------------")

    def close(self):
        self._writer.close()
