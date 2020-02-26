#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: zhangpengpeng
@Date:   2019-06-28
"""
import os
import sys
import random
from threading import Thread
from queue import Queue
from collections import namedtuple

import numpy as np
import tensorflow as tf

sys.path.append(".")

from model_optimize import optimize_graph
from tokenization import FullTokenizer, get_example 

random.seed(100)
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer(
    "task_index", None,
    "Specify the task index")

flags.DEFINE_integer(
    "task_num", None,
    "total task number")

flags.DEFINE_integer(
    "gpu_id", None,
    "gpu id")

flags.DEFINE_string("output_pattern", None, "Specify output file pattern, and pattern must contain %d")


class BertWorker(Thread):
  def __init__(self, index, graph_path, configuration):
    Thread.__init__(self)
    self._index = index
    self._graph_path = graph_path
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
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
    result = self.session.run(self.output_node, feed_dict={self.input_ids: features[0],
                                                           self.input_mask: features[1],
                                                           self.input_type_ids: features[2]})
    return np.argmax(result, axis=-1), np.exp(result)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def get_example(tokenizer, text_a, text_b, text_c, start_position, max_seq_length):
  if not max_seq_length or max_seq_length > 128:
    raise ValueError('max_seq_length must exist and be less than 128')
  tokens_a = tokenizer.tokenize(text_a)

  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)

  if text_c:
    tokens_c = tokenizer.tokenize(text_c)
    if start_position > 0:
      for index in range(start_position):
        token = tokens_b[index]
        if token=='[UNK]':
          tokens_b[index]='[MASK]'
        else:
          tokens_b[index] = token
    else:
      for index in range(len(tokens_c), len(tokens_b)):
        token = tokens_b[index]
        if token=='[UNK]':
          tokens_b[index]='[MASK]'
        else:
          tokens_b[index] = token
  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  return input_ids, input_mask, segment_ids


def main(_):
  import time
  os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
  current_path = os.path.dirname(os.path.abspath(__file__))
  tokenizer = FullTokenizer(os.path.join(current_path, './model/chinese_L-12_H-768_A-12/vocab.txt'))
  Configuration=namedtuple('Configuration', ['fp16', 'bert_config', 'checkpoint_path', 'graph_tmp_dir', 'max_seq_length'])

  fp16=False
  bert_config='./model/chinese_L-12_H-768_A-12/bert_config.json'
  #checkpoint_path='./model/ad/model_0622/model.ckpt-610194'
  checkpoint_path='./model/ad/model_pretrain_0924/model.ckpt-152548'
  graph_tmp_dir='./model/ad/tmp/'
  max_seq_length=70
  configuration=Configuration(fp16, bert_config, checkpoint_path,graph_tmp_dir,max_seq_length)

  graph_path, bert_config = optimize_graph(configuration)
  worker = BertWorker(0, graph_path, configuration)
  
  data_path='/data1/zhangpengpeng/ad_data/new_eval_ins2/v0805'

  start=time.time()
  for no in range(10):
    suffix=10000+no
    slice_path=os.path.join(data_path, 'slice_%s' % str(suffix)[1:])
    slice_output_path=os.path.join(data_path, 'slice_output_%s' % str(suffix)[1:])
    slice_output_file=tf.gfile.Open(slice_output_path, 'w')
    if not tf.gfile.Exists(slice_path):
      continue
    if tf.gfile.Exists(slice_output_path):
      continue
    print(slice_path, slice_output_path)
    count=0
    with tf.gfile.Open(slice_path, 'r') as f:
      input_ids_list=[]
      input_mask_list=[]
      segment_ids_list=[]
      rows=[]
      for index, line in enumerate(f):
        row=line.split('\t', 4)
        if len(row)!=5:
          continue
        text_a=row[3].strip()
        text_b=row[-1].strip()
        text_c=row[1].strip()
        start_position=0
        feature=get_example(tokenizer, text_a, text_b, text_c, start_position, 70)
        input_ids_list.append(feature[0])
        input_mask_list.append(feature[1])
        segment_ids_list.append(feature[2])
        rows.append(row)
        if len(input_ids_list) == 20:
          features=(input_ids_list, input_mask_list, segment_ids_list)
          tags, scores = worker.predict(features)
          for i in range(len(input_ids_list)):
            slice_output_file.write('%f\t%s\n' % (scores[i][1], '\t'.join(rows[i][:3])))
          input_ids_list=[]
          input_mask_list=[]
          segment_ids_list=[]
          rows=[]
        count+=1
      if len(rows)>0:
        features=(input_ids_list, input_mask_list, segment_ids_list)
        tags, scores = worker.predict(features)
        for i in range(len(input_ids_list)):
          slice_output_file.write('%f\t%s\n' % (scores[i][1], '\t'.join(rows[i][:3])))
        slice_output_file.close()
    end=time.time()
    print("filename: %s\tqps: %d" % (slice_path, count/(end-start)))


if __name__=='__main__':
  flags.mark_flag_as_required("task_index")
  flags.mark_flag_as_required("task_num")
  flags.mark_flag_as_required("gpu_id")
  tf.app.run()
