#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: 熊猫侠客
@Date:   2019-06-05
"""

import os
import random
import re
from collections import Counter
import tensorflow as tf

random.seed(12345)

root_path='../'
sentence_delimeter={'。', '！', '!', '？', '，', '：'}

def load_match():
  matches=[]
  with tf.gfile.Open('./matches.txt') as file:
    for line in file:
      line=''.join([segment for segment in line.split()])
      match=line.strip()
      if match:  
        matches.append(match)
  return matches


def text_iterator(filename):
  with tf.gfile.Open(filename, 'r') as f:
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
  return arr


def get_matches(filename):
  matches=set()
  file_iter=text_iterator(os.path.join(root_path, filename))
  count=0
  for line in file_iter:
    row=line.split('\t', 2)
    count+=1
    if len(row)!=3 or (not re.match('\d+', row[0])):
      continue
    main=row[1].strip()
    ad=row[2].strip()
    start_position=ad.find(main)
    if start_position == -1:
      continue
    arr=get_match(ad, main, start_position)
    for match in arr:
      match=match.strip()
      if match[0] in sentence_delimeter:
        match=match[1:]
      if match[-1] in sentence_delimeter:
        match=match[:-1]
      matches.add(match)
    if count % 1000000==0:
      print('process %d' % count)
  with tf.gfile.Open('./matches.txt', 'w') as match_file:
    for match in matches:
      match_file.write('%s\n' % match)


def generate_dataset():
  match_counter=Counter()
  matches=load_match()
  matches_set=set(matches)

  split_pattern='|'.join(sentence_delimeter)
  pattern=re.compile(split_pattern)

  train_examples=[]
  filenames=tf.gfile.ListDirectory(os.path.join(root_path, 'idea_ctr'))
  
  counter=Counter()
  for filename in filenames:
    iter = text_iterator(os.path.join(root_path, 'idea_ctr', filename))
    for line in iter:
      row=line.strip().split('\t')
      if not row:
        continue
      segments=re.split(pattern, row[-2])
      segments=[segment for segment in segments if segment]
      if len(segments) <= 1:
        continue

      ad=''.join([s for s in row[-2].split()])
      for seg_index, segment in enumerate(segments):
        if segment not in matches_set:
          segment=''.join([s for s in segment.split()])
          matches.append(segment)
          matches_set.add(segment)

        start_position=ad.find(segment)
        train_examples.append((1, seg_index, segment, ad.replace(segment, '“' * len(segment))))
        counter[1]+=1
        for index in range(3):
          random_index=random.randint(0, len(matches)-1)
          tmp=matches[random_index]
          if segment!=tmp:
            train_examples.append((0, seg_index, tmp, ad.replace(segment, '“' * len(tmp))))
            counter[0]+=1
            break

  ins_iter=text_iterator(os.path.join(root_path, 'ins.all'))
  for line in ins_iter:
    ad=line.strip()
    if not line:
      continue
    line=''.join([segment for segment in line.split()])
    segments=re.split(pattern, line)
    segments=[segment for segment in segments if segment]
    if len(segments) <= 1:
      continue
    ad=line
    for seg_index, segment in enumerate(segments):
      if segment not in matches_set:
        matches.append(segment)
        matches_set.add(segment)
      start_position=ad.find(segment)
      train_examples.append((1, seg_index, segment, ad.replace(segment, '“' * len(segment))))
      counter[1]+=1
      for index in range(3):
        random_index=random.randint(0, len(matches)-1)
        tmp=matches[random_index]
        if segment!=tmp:
          train_examples.append((0, seg_index, tmp, ad.replace(segment, '“' * len(tmp))))
          counter[0]+=1
          break
  train_counter=Counter()
  with tf.gfile.Open('train_0620.txt', 'w') as train_file,\
    tf.gfile.Open('test_0620.txt', 'w') as test_file:
    for t in train_examples:
      if random.random() <= 0.8:
        train_file.write('%d\t%d\t%s\t%s\n' % t)
        train_counter[t[0]]+=1
      else:
        test_file.write('%d\t%d\t%s\t%s\n' % t)
  print(train_counter)


def tokenize(filename):
  import jieba
  jieba.load_userdict('match_dict.dic')
  file_iter=text_iterator(filename)
  for line in file_iter:
    if line:
      tokens=[token[0] for token in jieba.cut(line)]
      print(' '.join(tokens))

if __name__=='__main__':
  generate_dataset()
