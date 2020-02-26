#!/usr/bin/python3
# coding:utf-8
import os, re

import tensorflow as tf

root_path='./'
sentence_delimeter={'。', '！', '!', '？', '，', '：'}


def text_iterator(filename):
  with tf.gfile.Open(os.path.join(root_path, filename), 'r') as f:
    for line in f:
      line=line.strip()
      if line:
        yield line


def get_match(ad, main, start_position):
  prefix=ad[0: start_position]
  suffix=ad[start_position+len(main):]
  pre=prefix
  arr=[]
  if len(prefix) > 0:
    arr.append(prefix)
  if len(suffix) > 0:
    arr.append(suffix)
  for index, match in enumerate(arr):
    match=match.strip()
    if match[0] in sentence_delimeter:
      match=match[1:]
    if match[-1] in sentence_delimeter:
      match=match[:-1]
    arr[index]=match
  return arr


def get_eval_dataset(filename, count=1000000):
  file_iter=text_iterator(filename)

  split_pattern='|'.join(sentence_delimeter)
  pattern=re.compile(split_pattern)

  current_count=0
  for line in file_iter:
    row=line.split('\t', 2)
    if len(row)!=3:
      continue
    ad=row[-1].strip()
    main=row[-2].strip()
    start_position=ad.find(main)
    matches=get_match(ad, main, start_position)

    segments=re.split(pattern, ad)
    segments=[segment for segment in segments if segment]
    segment_index=0
    if start_position==0:
      segment_index=len(segments)-1
    else:
      segment_index=0
    current_segment_index=0
    tmp_ad=[]
    for ch in ad:
      if ch in sentence_delimeter:
        tmp_ad.append(ch)
        current_segment_index+=1
        continue
      if current_segment_index==segment_index:
        tmp_ad.append("“")
      else:
        tmp_ad.append(ch)
    print('%d\t%d\t%s\t%s\t%s' % (0, segment_index, matches[0], ''.join(tmp_ad), ad))

    current_count+=1
    if count < current_count:
      break


def eval_dataset(filename, predict_filename):
  file_iter=text_iterator(filename)
  predict_iter=text_iterator(predict_filename)
  examples=[]
  for line in file_iter:
    examples.append(line)
  predicts=[]
  for predict in predict_iter:
    predicts.append(predict)
  print(len(examples), len(predicts))
  for index in range(len(predicts)):
    print('%s\t%s' % (examples[index], predicts[index]))


if __name__=='__main__':
  #get_eval_dataset('rewrite_ins2')
  eval_dataset('eval_0620.txt', 'predict_pretrain_0801.txt')
