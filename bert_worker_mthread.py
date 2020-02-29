#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: zhangpengpeng
@Date:   2019-04-24
"""
import sys
from threading import Thread
from queue import Queue

import numpy as np
import tensorflow as tf

sys.path.append(".")

from server.model_optimize import optimize_graph

class BertWorker(Thread):
  def __init__(self, index, graph_path, configuration):
    Thread.__init__(self)
    self._index = index
    self._graph_path = graph_path
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=configuration.per_process_gpu_memory_fraction,
                                visible_device_list='%d' % index)
    tf_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
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

class ThreadPool:
  """ Pool of threads consuming tasks from a queue """

  def __init__(self, num_threads, graph_path, configuration):
    self.tasks = Queue(num_threads)
    for index in range(num_threads):
      worker = BertWorker(int(index / 3), graph_path, configuration)
      worker.start()
      self.tasks.put(worker)

  def add_task(self, value):
    """ Add a task to the queue """
    self.tasks.put((value))

  def get(self):
    """ Add a task to the queue """
    worker = self.tasks.get(block=True)
    return worker

  def put(self, worker):
    self.tasks.put(worker, block=True)

  def wait_completion(self):
    """ Wait for completion of all the tasks in the queue """
    self.tasks.join()

  @classmethod
  def get_threadpool(self, config):
    graph_path, bert_config = optimize_graph(config)

    bert_worker_pool = ThreadPool(num_threads=2, graph_path=graph_path, configuration=config)

    return bert_worker_pool


class Inference(Thread):
  def __init__(self, name, bert_worker_pool, tokenizer):
    Thread.__init__(self, name=name)
    self._bert_worker_pool = bert_worker_pool
    self._tokenizer = tokenizer
    self._name = name

  def run(self):
    category_correct_counter = Counter()
    category_total_counter = Counter()
    start = time.time()
    with tf.gfile.Open(os.path.join('../data/data.txt')) as file:
      correct_count = 0
      total_count = 0
      for line in file:
        line = line.strip()
        arr = line.split('\t', 3)
        if len(arr) != 4:
          continue
        total_count += 1
        category = arr[-3]
        label = int(arr[-2])
        sentence = arr[-1]
        example = get_example(self._tokenizer, text_a=sentence, text_b=None, max_seq_length=70)

        worker = self._bert_worker_pool.get()
        result = worker.predict(example)
        self._bert_worker_pool.put(worker)

        category_total_counter[category] += 1
        if result[0][0] == label:
          correct_count += 1
          category_correct_counter[category] += 1
      end = time.time()
      print(float(total_count) / float(end - start))
      print(correct_count, total_count)
      print('accuracy: %f' % (float(correct_count) / float(total_count)))
      for key, count in category_total_counter.items():
        print(key, float(category_correct_counter[key]) / float(category_total_counter[key]))


if __name__=='__main__':
  import os, time
  from collections import Counter
  from server.tokenization import FullTokenizer, get_example

  from common import config

  current_path = os.path.dirname(os.path.abspath(__file__))
  tokenizer = FullTokenizer(os.path.join(current_path, '../vocab.txt'))

  graph_path, bert_config = optimize_graph(config)

  bert_worker_pool = ThreadPool(num_threads=1, graph_path=graph_path, configuration=config)

  thread1 = Inference('thread1', bert_worker_pool, tokenizer)
  thread2 = Inference('thread2', bert_worker_pool, tokenizer)
  thread1.start()
  thread2.start()
  time.sleep(10000)
