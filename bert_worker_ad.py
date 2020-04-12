#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: 熊猫侠客
@Date:   2019-05-20
"""
import os
import sys
from threading import Thread
from queue import Queue
from collections import namedtuple

import numpy as np
import tensorflow as tf

sys.path.append(".")

from model_optimize import optimize_graph
from tokenization import FullTokenizer, get_example 

class BertWorker(Thread):
  def __init__(self, index, graph_path, configuration):
    Thread.__init__(self)
    self._index = index
    self._graph_path = graph_path
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    tf_config = tf.ConfigProto(gpu_options=gpu_options,
                               inter_op_parallelism_threads=4,
                               intra_op_parallelism_threads=3,
                               allow_soft_placement=True)
    self.session = tf.InteractiveSession(config=tf_config)
    self._model_prefix = 'scope%d' % index
    with tf.name_scope(self._model_prefix):
      self._load_model(graph_path)

  def _load_model(self, graph_path):
    with tf.gfile.GFile(graph_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    prefix = 'import%d' % self._index
    with tf.device('/gpu:0'):
      tf.import_graph_def(graph_def, name=prefix, return_elements=['loss/LogSoftmax:0'])

    self.output_node = self.session.graph.get_tensor_by_name('%s/%s/loss/LogSoftmax:0' % (self._model_prefix, prefix))
    self.input_ids = self.session.graph.get_tensor_by_name('%s/%s/input_ids:0' % (self._model_prefix, prefix))
    self.input_mask = self.session.graph.get_tensor_by_name('%s/%s/input_mask:0' % (self._model_prefix, prefix))
    self.input_type_ids = self.session.graph.get_tensor_by_name('%s/%s/input_type_ids:0' % (self._model_prefix, prefix))

  def predict(self, features):
    result = self.session.run(self.output_node, feed_dict={self.input_ids: [features[0]],
                                                           self.input_mask: [features[1]],
                                                           self.input_type_ids: [features[2]]})
    return np.argmax(result, axis=-1), np.exp(result)


if __name__=='__main__':
  import time
  current_path = os.path.dirname(os.path.abspath(__file__))
  tokenizer = FullTokenizer(os.path.join(current_path, './model/chinese_L-12_H-768_A-12/vocab.txt'))
  Configuration=namedtuple('Configuration', ['fp16', 'bert_config', 'checkpoint_path', 'graph_tmp_dir'])

  fp16=False
  bert_config='./model/chinese_L-12_H-768_A-12/bert_config.json'
  checkpoint_path='./model/ad/model_0605/model.ckpt-366269'
  graph_tmp_dir='./model/ad/tmp/'
  configuration=Configuration(fp16, bert_config, checkpoint_path,graph_tmp_dir)

  graph_path, bert_config = optimize_graph(configuration)
  worker = BertWorker(0, graph_path, configuration)
  text_a="拿豪礼"
  text_b="龙虾种苗基地,送苗上门,免费技术指导，“““"
  feature = get_example(tokenizer, text_a=text_a, text_b=text_b, max_seq_length=60)
  tag, score = worker.predict(feature)
  print('%f\t%s\t%s\n' % (score[0][1], text_a, text_b))
