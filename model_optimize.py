#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: 熊猫侠客
@Date:   2019-05-20
"""
import os, sys, re, collections
import json
import tempfile
import logging

sys.path.append('.')
import modeling as modeling


logger = logging.Logger(name='graph optimize logger', level=logging.DEBUG)


def import_tf(verbose=False, use_fp16=False):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
  os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
  os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
  import tensorflow as tf
  tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
  return tf


tf = import_tf(verbose=logging.DEBUG, use_fp16=True)


def optimize_graph(configuration):
  try:
    # we don't need GPU for optimizing the graph
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

    tf_config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

    if configuration.fp16:
      logger.warning('fp16 is turned on! '
                     'Note that not all CPU GPU support fast fp16 instructions, '
                     'worst case you will have degraded performance!')
    logger.info('model config: %s' % configuration.bert_config)

    with tf.gfile.GFile(configuration.bert_config, 'r') as f:
      bert_config = modeling.BertConfig.from_dict(json.load(f))
    
    max_seq_lengh=configuration.max_seq_length

    logger.info('build graph...')
    # input placeholders, not sure if they are friendly to XLA
    input_ids = tf.placeholder(tf.int32, (None, max_seq_lengh), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None, max_seq_lengh), 'input_mask')
    input_type_ids = tf.placeholder(tf.int32, (None, max_seq_lengh), 'input_type_ids')

    # jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    # with jit_scope():
    input_tensors = [input_ids, input_mask, input_type_ids]

    model = modeling.BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=input_type_ids,
      use_one_hot_embeddings=False)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [2], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      logits = tf.matmul(output_layer, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      output_tensors=[log_probs]

    tvars = tf.trainable_variables()

    (assignment_map, initialized_variable_names
     ) = get_assignment_map_from_checkpoint(tvars, configuration.checkpoint_path)

    tf.train.init_from_checkpoint(configuration.checkpoint_path, assignment_map)
    tmp_g = tf.get_default_graph().as_graph_def()

    with tf.Session(config=tf_config) as sess:
      logger.info('load parameters from checkpoint...')

      sess.run(tf.global_variables_initializer())
      dtypes = [n.dtype for n in input_tensors]
      logger.info('optimize...')
      tmp_g = optimize_for_inference(
        tmp_g,
        [n.name[:-2] for n in input_tensors],
        [n.name[:-2] for n in output_tensors],
        [dtype.as_datatype_enum for dtype in dtypes],
        False)

      logger.info('freeze...')
      tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors],
                                             use_fp16=configuration.fp16)

    tmp_file = tempfile.NamedTemporaryFile('w', delete=False, dir=configuration.graph_tmp_dir).name
    logger.info('write graph to a tmp file: %s' % tmp_file)
    with tf.gfile.GFile(tmp_file, 'wb') as f:
      f.write(tmp_g.SerializeToString())
    return tmp_file, bert_config
  except Exception:
    logger.error('fail to optimize the graph!', exc_info=True)


def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None,
                                   use_fp16=False):
  from tensorflow.python.framework.graph_util_impl import extract_sub_graph
  from tensorflow.core.framework import graph_pb2
  from tensorflow.core.framework import node_def_pb2
  from tensorflow.core.framework import attr_value_pb2
  from tensorflow.core.framework import types_pb2
  from tensorflow.python.framework import tensor_util

  def patch_dtype(input_node, field_name, output_node):
    if use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT):
      output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

  inference_graph = extract_sub_graph(input_graph_def, output_node_names)

  variable_names = []
  variable_dict_names = []
  for node in inference_graph.node:
    if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
      variable_name = node.name
      if ((variable_names_whitelist is not None and
           variable_name not in variable_names_whitelist) or
        (variable_names_blacklist is not None and
         variable_name in variable_names_blacklist)):
        continue
      variable_dict_names.append(variable_name)
      if node.op == "VarHandleOp":
        variable_names.append(variable_name + "/Read/ReadVariableOp:0")
      else:
        variable_names.append(variable_name + ":0")
  if variable_names:
    returned_variables = sess.run(variable_names)
  else:
    returned_variables = []
  found_variables = dict(zip(variable_dict_names, returned_variables))

  output_graph_def = graph_pb2.GraphDef()
  how_many_converted = 0
  for input_node in inference_graph.node:
    output_node = node_def_pb2.NodeDef()
    if input_node.name in found_variables:
      output_node.op = "Const"
      output_node.name = input_node.name
      dtype = input_node.attr["dtype"]
      data = found_variables[input_node.name]

      if use_fp16 and dtype.type == types_pb2.DT_FLOAT:
        output_node.attr["value"].CopyFrom(
          attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(data.astype('float16'),
                                                 dtype=types_pb2.DT_HALF,
                                                 shape=data.shape)))
      else:
        output_node.attr["dtype"].CopyFrom(dtype)
        output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
          tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type,
                                               shape=data.shape)))
      how_many_converted += 1
    elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
      # placeholder nodes
      output_node.op = "Identity"
      output_node.name = input_node.name
      output_node.input.extend([input_node.input[0]])
      output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
      if "_class" in input_node.attr:
        output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
    else:
      # mostly op nodes
      output_node.CopyFrom(input_node)

    patch_dtype(input_node, 'dtype', output_node)
    patch_dtype(input_node, 'T', output_node)
    patch_dtype(input_node, 'DstT', output_node)
    patch_dtype(input_node, 'SrcT', output_node)
    patch_dtype(input_node, 'Tparams', output_node)

    if use_fp16 and ('value' in output_node.attr) and (
      output_node.attr['value'].tensor.dtype == types_pb2.DT_FLOAT):
      # hard-coded value need to be converted as well
      output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
          output_node.attr['value'].tensor.float_val[0],
          dtype=types_pb2.DT_HALF)))

    output_graph_def.node.extend([output_node])

  output_graph_def.library.CopyFrom(inference_graph.library)

  output_graph_def=tf.graph_util.remove_training_nodes(output_graph_def)
  return output_graph_def


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
