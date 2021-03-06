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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import random

import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("slice_name", 'slice_1', "output name prefix")

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("root_data_path", None,
                    "specify data save path")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances_list, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for instances in instances_list:
    for (inst_index, instance) in enumerate(instances):
      input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
      input_mask = [1] * len(input_ids)
      segment_ids = list(instance.segment_ids)
      assert len(input_ids) <= max_seq_length

      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      masked_lm_positions = list(instance.masked_lm_positions)
      masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
      masked_lm_weights = [1.0] * len(masked_lm_ids)

      while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)
      
      next_sentence_label = 1 if instance.is_random_next else 0


      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(input_ids)
      features["input_mask"] = create_int_feature(input_mask)
      features["segment_ids"] = create_int_feature(segment_ids)
      features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
      features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
      features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
      features["next_sentence_labels"] = create_int_feature([next_sentence_label])

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))

      writers[writer_index].write(tf_example.SerializeToString())
      writer_index = (writer_index + 1) % len(writers)

      total_written += 1

      if inst_index < 10:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in instance.tokens]))

        for feature_name in features.keys():
          feature = features[feature_name]
          values = []
          if feature.int64_list.value:
            values = feature.int64_list.value
          elif feature.float_list.value:
            values = feature.float_list.value
          tf.logging.info(
              "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  sentences = []
  count = 0
  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line=reader.readline()
        if not line:
          break
        line = tokenization.convert_to_unicode(line)
        line = line.strip()

        if line:
          row=line.split('\t', 3)
          if len(row)==4:
            tokens_a = tokenizer.tokenize(row[2].strip())
            tokens_b = tokenizer.tokenize(row[3].strip())
            if len(tokens_a) > 1 and len(tokens_a) < len(tokens_b):
              row[2]=tokens_a
              row[3]=tokens_b
              sentences.append(row)
              count+=1
          if count > 0 and count % 500000 == 0:
            tf.logging.info("*** Reading %d from input files ***" % count)
            # Remove empty sentences
            sentences = [x for x in sentences if x]
            vocab_words = list(tokenizer.vocab.keys())
            for i in range(dupe_factor):
              instances = []
              print('dupe factor: %d' % i)
              instances.extend(
                create_instances_from_sentences(
                      sentences, max_seq_length, short_seq_prob,
                      masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
              print('shuffle instaneces: %d' % len(instances))
              rng.shuffle(instances)
              yield instances
            sentences = []

  tf.logging.info("***Last batch %d from input files ***" % len(sentences))
  # Remove empty sentences
  sentences = [x for x in sentences if x]
  print('sentences %d' % len(sentences))
  tf.logging.info("*** sentences count: %d ***" % len(sentences))
  vocab_words = list(tokenizer.vocab.keys())
  for _ in range(dupe_factor):
    instances = []
    instances.extend(
      create_instances_from_sentences(
        sentences, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    yield instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def create_instances_from_sentences(
    sentences, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for sentences."""

  # Account for [CLS], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  print("create masked tokens")
  #create masked tokens
  instances = []
  i = 0
  while i < len(sentences):
    row = sentences[i]
    tokens_a = []
    for j in range(len(row[2])):
      tokens_a.append(row[2][j])
    tokens_b = []
    for j in range(len(row[3])):
      tokens_b.append(row[3][j])

    tokens_b=tokens_b[:max_num_tokens-len(tokens_a)]
    
    if len(tokens_a) >= len(tokens_b):
      i+=1
      continue

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions(
         tokens, tokens_a, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
    instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels,
        is_random_next=bool(int(row[0].strip())))
    instances.append(instance)
    i += 1
    if i % 10000==0:
      print("create masked instances: %d" % len(instances))

  return instances


def create_masked_lm_predictions(tokens, tokens_a, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""
  tokens_a_len=len(tokens_a)
  start_index=random.randint(tokens_a_len+2, len(tokens)-tokens_a_len-1)

  cand_indexes = []
  for i in range(start_index, start_index+tokens_a_len):
    if i < (len(tokens)-1):
      cand_indexes.append(i)
 
  output_tokens = list(tokens)

  masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  # pylint: disable=invalid-name

  masked_lms = []
  for index in cand_indexes:
    if len(masked_lms)>=max_predictions_per_seq:
      break
    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(masked_lm(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return output_tokens, masked_lm_positions, masked_lm_labels


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = [os.path.join(FLAGS.root_data_path, '%s_%s' % (FLAGS.slice_name, file)) for file in
                  FLAGS.output_file.split(",")]
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("root_data_path")
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
