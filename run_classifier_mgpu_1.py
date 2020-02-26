# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os,json, random
import numpy as np
import modeling
import optimization
import tokenization
import tensorflow as tf

random.seed(100)
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run predict on the dev set.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("n_gpus", 1, "How many gpu to use.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_json(cls, input_file):
    """Read json file"""
    with tf.gfile.Open(input_file, "r") as f:
      lines = []
      for line in f:
        if line.strip() is not None:
          lines.append(json.loads(line))
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % i
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class SentimentProcessor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.txt")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      comment = line['comment']
      half = int(np.floor(len(comment) / 2))
      text_a = tokenization.convert_to_unicode("".join(comment[0:half]))
      text_b = tokenization.convert_to_unicode("".join(comment[half:]))
      label = str(line['items'][0]['sentiment'])
      if label == "2":
        label = tokenization.convert_to_unicode("1")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      elif label == "0":
        label = tokenization.convert_to_unicode("0")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class WOS5736Processor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.txt")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      comment = line[1]
      sentences = comment.split(',')
      half = int(np.floor(len(sentences) / 2))
      text_a = tokenization.convert_to_unicode(",".join(sentences[0:half]))
      text_b = tokenization.convert_to_unicode("".join(sentences[half:]))

      label = tokenization.convert_to_unicode(line[0])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class ToutiaoProcessor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.txt")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      sentences = line[1]
      half = int(np.floor(len(sentences) / 2))
      text_a = tokenization.convert_to_unicode(sentences)

      label = tokenization.convert_to_unicode(line[0])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    return examples


class TitleAuditProcessor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train_all_0424.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test_pinggu_0426.tsv")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    size = len(lines)
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label = tokenization.convert_to_unicode(line[2].strip())
      title = tokenization.convert_to_unicode(line[-1].lower().strip())

      # examples.append(
      #   InputExample(guid=guid, text_a='+%s' % category, text_b=title, label=label))
      examples.append(
        InputExample(guid=guid, text_a=title, text_b=None, label=label))

    if set_type == 'train':
      random.shuffle(examples)

    return examples


class AbuseProcessor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train_abuse_0531.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "eval_data_revise.txt")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    size = len(lines)
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label = tokenization.convert_to_unicode(line[2].strip())
      title = tokenization.convert_to_unicode(line[-1].lower().strip())
      if set_type=='dev':
        label="0"

      # examples.append(
      #   InputExample(guid=guid, text_a='+%s' % category, text_b=title, label=label))
      examples.append(
        InputExample(guid=guid, text_a=title, text_b=None, label=label))

    if set_type == 'train':
      random.shuffle(examples)

    return examples


class AdLmProcessor(DataProcessor):

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train_0610_1.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "eval_0605.tsv")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []

    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      if set_type=='dev':
        text_a = tokenization.convert_to_unicode(line[1])
        test_b = tokenization.convert_to_unicode(line[2])
        label='1'
      else:
        text_a = tokenization.convert_to_unicode(line[2])
        test_b = tokenization.convert_to_unicode(line[-1])
        label=tokenization.convert_to_unicode(line[1])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=test_b, label=label))

    return examples


class HighLevelProcessor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train_technology_0604.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test_technology_0604.tsv")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []

    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      category = tokenization.convert_to_unicode(line[1])
      title = tokenization.convert_to_unicode(line[-1].lower())
      label = line[2]
      examples.append(
        InputExample(guid=guid, text_a=title, text_b=None, label=label))

    return examples


class WOS46985Processor(DataProcessor):
  """Processor for the Sentiment data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.txt")), "dev")

  def get_labels(self):
    """See base class."""
    return [str(num) for num in range(134)]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      comment = line[1]
      sentences = comment.split(',')
      half = int(np.floor(len(sentences) / 2))
      text_a = tokenization.convert_to_unicode(",".join(sentences[0:half]))
      text_b = tokenization.convert_to_unicode("".join(sentences[half:]))

      label = tokenization.convert_to_unicode(line[0])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_file):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
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

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("guid: %s" % (example.guid))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature([label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32, name="output_labels")

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits)


def _deduplicate_indexed_slices(values, indices):
  """Sums `values` associated with any non-unique `indices`.
  Args:
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
    dimension of `values` (as in an IndexedSlices object).
  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  unique_indices, new_index_positions = tf.unique(indices)
  summed_values = tf.unsorted_segment_sum(
    values, new_index_positions,
    tf.shape(unique_indices)[0])
  return summed_values, unique_indices


def average_gradients(tower_grads, batch_size, options):
  # calculate average gradient for each shared variable across all GPUs
  average_grads = []
  count = 0
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    # We need to average the gradients across each GPU.
    count += 1
    g0, v0 = grad_and_vars[0]

    if g0 is None:
      # no gradient for this variable, skip it
      average_grads.append((g0, v0))
      continue

    if isinstance(g0, tf.IndexedSlices):
      # If the gradient is type IndexedSlices then this is a sparse
      #   gradient with attributes indices and values.
      # To average, need to concat them individually then create
      #   a new IndexedSlices object.
      indices = []
      values = []
      for g, v in grad_and_vars:
        indices.append(g.indices)
        values.append(g.values)
      all_indices = tf.concat(indices, 0)
      avg_values = tf.concat(values, 0) / len(grad_and_vars)
      # deduplicate across indices
      av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
      grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

    else:
      # a normal tensor can just do a simple average
      grads = []
      for g, v in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)
        # Append on a 'tower' dimension which we will average over
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)

    # the Variables are redundant because they are shared
    # across towers. So.. just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)

    average_grads.append(grad_and_var)

  assert len(average_grads) == len(list(zip(*tower_grads)))

  return average_grads


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
  # wrapper around tf.clip_by_global_norm that also does summary ops of norms

  # compute norms
  # use global_norm with one element to handle IndexedSlices vs dense
  norms = [tf.global_norm([t]) for t in t_list]

  # summary ops before clipping
  summary_ops = []
  for ns, v in zip(norms, variables):
    name = 'norm_pre_clip/' + v.name.replace(":", "_")
    summary_ops.append(tf.summary.scalar(name, ns))

  # clip
  clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

  # summary ops after clipping
  norms_post = [tf.global_norm([t]) for t in clipped_t_list]
  for ns, v in zip(norms_post, variables):
    name = 'norm_post_clip/' + v.name.replace(":", "_")
    summary_ops.append(tf.summary.scalar(name, ns))

  summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

  return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, all_clip_norm_val, do_summaries, global_step):
  # grads = [(grad1, var1), (grad2, var2), ...]
  def _clip_norms(grad_and_vars, val, name):
    # grad_and_vars is a list of (g, v) pairs
    grad_tensors = [g for g, v in grad_and_vars]
    vv = [v for g, v in grad_and_vars]
    scaled_val = val
    if do_summaries:
      clipped_tensors, g_norm, so = clip_by_global_norm_summary(
        grad_tensors, scaled_val, name, vv)
    else:
      so = []
      clipped_tensors, g_norm = tf.clip_by_global_norm(
        grad_tensors, scaled_val)

    ret = []
    for t, (g, v) in zip(clipped_tensors, grad_and_vars):
      ret.append((t, v))

    return ret, so

  ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

  assert len(ret) == len(grads)

  return ret, summary_ops


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, n_gpus, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    num_gpus = n_gpus
    if is_training:
      optimizer = optimization.create_optimizer_mgpu(learning_rate, num_train_steps, num_warmup_steps)
    else:
      num_gpus=1

    input_ids_list = tf.split(features["input_ids"], num_or_size_splits=num_gpus, axis=0)
    input_mask_list = tf.split(features["input_mask"], num_or_size_splits=num_gpus, axis=0)
    segment_ids_list = tf.split(features["segment_ids"], num_or_size_splits=num_gpus, axis=0)
    label_ids_list = tf.split(features["label_ids"], num_or_size_splits=num_gpus, axis=0)

    tower_grads = []
    train_perplexity = 0
    for index in range(num_gpus):
      with tf.name_scope('replica_%d' % index):
        with tf.device('/gpu:%d' % index):
          (total_loss, per_example_loss, logits) = create_model(
              bert_config, is_training,
              input_ids_list[index], input_mask_list[index], segment_ids_list[index], label_ids_list[index],
              num_labels, use_one_hot_embeddings)

          tvars = tf.trainable_variables()

          scaffold_fn = None
          if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(
                 tvars, init_checkpoint)
            for var in tvars:
              param_name = var.name[:-2]
              tf.get_variable(
                name=param_name + "/adam_m",
                shape=var.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
              tf.get_variable(
                name=param_name + "/adam_v",
                shape=var.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            if use_tpu:
              def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

              scaffold_fn = tpu_scaffold
            else:
              tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

          tf.logging.info("**** Trainable Variables ****")
          tf.logging.info('device: %d init' % index)
          if index == 0:
            for var in tvars:
              init_string = ""
              if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
              tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                              init_string)
          if is_training:
            # reuse variables
            tf.get_variable_scope().reuse_variables()
            loss = total_loss
            # get gradients
            grads = optimizer.compute_gradients(
              loss,
              aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
            )
            tower_grads.append(grads)
            # keep track of loss across all GPUs
            train_perplexity += loss

    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      new_global_step = global_step + 1

      average_grads = average_gradients(tower_grads, None, None)
      #average_grads, norm_summary_ops = clip_grads(average_grads, 1.0, True, global_step)
      train_op = optimizer.apply_gradients(average_grads)
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=train_perplexity / n_gpus,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              'predictions': predictions,
          })
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids_list[0], logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, batch_size, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    # batch_size = params["batch_size"]
    # batch_size=8
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=10000)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "sentiment": SentimentProcessor,
      "wos5736": WOS5736Processor,
      "wos46985": WOS46985Processor,
      "toutiao": ToutiaoProcessor,
      "title_audit": TitleAuditProcessor,
      "high_level": HighLevelProcessor,
      "abuse": AbuseProcessor,
      "ad": AdLmProcessor,
  }

  #if not FLAGS.do_train and not FLAGS.do_eval:
    #raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / (FLAGS.train_batch_size * FLAGS.n_gpus) * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      n_gpus=FLAGS.n_gpus,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size * FLAGS.n_gpus,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    convert_examples_to_features(train_examples, label_list,
                                 FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size * FLAGS.n_gpus)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        batch_size=FLAGS.train_batch_size * FLAGS.n_gpus,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    predict_examples = processor.get_dev_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    convert_examples_to_features(predict_examples, label_list,
                                 FLAGS.max_seq_length, tokenizer, predict_file)

    tf.logging.info("***** Running prediction *****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        drop_remainder=True)

    output_predict_file = os.path.join(FLAGS.output_dir, "predict.txt")
    with tf.gfile.Open(output_predict_file, 'w') as file:
      for result in estimator.predict(predict_input_fn):
        file.write('%d\n' % result['predictions'])

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    convert_examples_to_features(eval_examples, label_list,
                                 FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    # if FLAGS.use_tpu:
    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
