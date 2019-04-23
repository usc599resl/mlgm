from datetime import datetime
import os

import tensorflow as tf
from tensorflow import summary as summ


class Logger:
    """
    A visualization module uses TensorBoard to log the experiment info.
    
    :params exp_name: str, name for the experiment.
    :params graph: tf.Graph, the computation graph of the model.
    :params save_period: int, the frequency of flush the logger to disk.
    :params std_out_period: int, the frequency of flush the logger to std output.
    """
    def __init__(self, exp_name, graph=None, save_period=50, std_out_period=50):
        self._log_path = 'data/{0}/{0}_{1}'.format(exp_name, datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f'))
        self._writer = summ.FileWriter(self._log_path, graph)
        self._summary_mrg = None
        self._writer.flush()
        self._save_period = int(save_period)
        self._std_out_period = int(std_out_period)

    def new_summary(self):
        """Create a new tf.Summary."""
        self._summary = tf.Summary()
        self._std_out = {}

    def add_value(self, name, value):
        """Add key-value pairs to the summary writer."""
        if isinstance(value, list):
            for i, val in enumerate(value):
                name_id = name + '{}'.format(i)
                self._summary.value.add(tag=name_id, simple_value=val)
                self._std_out.update({name_id: val})
        else:
            self._summary.value.add(tag=name, simple_value=value)
            self._std_out.update({name: value})

    def add_graph(self, graph):
        """Add graph to the summary writer."""
        self._writer.add_graph(graph)

    def add_image(self, image, itr):
        """Add image to the summary writer."""
        self._writer.add_summary(image, itr)

    @property
    def summary(self):
        return self._summary_mrg

    def save_tf_variables(self, var_list, itr, sess):
        if not (itr % self._save_period) and itr > 0:
            saver = tf.train.Saver(var_list)
            saver.save(sess, self._log_path + '/itr_{}'.format(itr))

    def dump_summary(self, itr):
        self._writer.add_summary(self._summary, itr)
        self._writer.flush()
        if not (itr % self._std_out_period) and itr > 0:
            print('--------------------------------------------------')
            print('exp_name: {}'.format(self._log_path))
            print('itr: {}'.format(itr))
            for k, v in self._std_out.items():
                print('{}: {}'.format(k, v))
            print('--------------------------------------------------')

    def close(self):
        """Close the summray writer."""
        self._writer.close()
