#!/usr/bin/env python
# encoding=utf-8
'''
Author: 	Zhiyang.zzy 
Date: 		2020-09-23 22:46:15
Contact: 	Zhiyangchou@gmail.com
FilePath: /ChineseNLU/data.py
Desc: 		catslu数据处理，作为数据输入格式。
todo: 
- 纯实体
- 
'''
# here put the import lib
from inspect import isbuiltin
import json
from os import read
import random
from tensorflow.python.keras.backend import dtype
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import sequence
import math
import numpy as np
import os 
from util import read_file, convert_to_unicode, read_json

def parse_smp2020_json_file(file_):
    """
    desc:将json转为数组，然后将其返回
    return: out_arr=[[query, domain, intent, slots], ], slots格式：[slot_k, slot_v]
    """
    data_set = {}
    with open(file_) as f:
        data_set = json.load(f)
        # for each in data_set:
        # each = {"text":, "domain":, "intent":, "slots": {k:v}}
    return data_set

def get_smp2020(file_dir):
    """
    smp2020数据，统计di意图个数，每个意图的样本数目
    todo: FewJoint 所有的数据合并在一起，重新划分测试集和训练，每个意图划分比例8：1：1
    train: 5024, dev: , test:
    """
    train_file = file_dir + "/train/source.json"
    parse_smp2020_json_file(train_file)

def get_train_catslu_data(file_):
    print("test")
    with open(file_) as f:
        data_set = json.load(f)
    
    pass

class Vocabulary(object):
    def __init__(self, meta_file=None, allow_unk=0, unk="$UNK$", pad="$PAD$", max_len=None):
        self.voc2id = {}
        self.id2voc = {}
        self.unk = unk
        self.pad = pad
        self.max_len = max_len
        self.allow_unk = allow_unk
        if meta_file:
            with open(meta_file, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = convert_to_unicode(line.strip("\n"))
                    self.voc2id[line] = i
                    self.id2voc[i] = line
            self.size = len(self.voc2id)
            self.oov_num = self.size + 1

    def fit(self, words_list):
        """
        :param words_list: [[w11, w12, ...], [w21, w22, ...], ...]
        :return:
        """
        word_lst = []
        word_lst_append = word_lst.append
        for words in words_list:
            if not isinstance(words, list):
                print(words)
                continue
            for word in words:
                word = convert_to_unicode(word)
                word_lst_append(word)
        word_counts = Counter(word_lst)
        if self.max_num_word < 0:
            self.max_num_word = len(word_counts)
        sorted_voc = [w for w, c in word_counts.most_common(self.max_num_word)]
        self.max_num_word = len(sorted_voc)
        self.oov_index = self.max_num_word + 1
        self.voc2id = dict(zip(sorted_voc, range(1, self.max_num_word + 1)))
        return self

    def _transform2id(self, word):
        word = convert_to_unicode(word)
        if word in self.voc2id:
            return self.voc2id[word]
        elif self.allow_unk:
            return self.voc2id[self.unk]
        else:
            print(word)
            raise ValueError("word:{} Not in voc2id, please check".format(word))

    def _transform_seq2id(self, words, padding=0):
        out_ids = []
        words = convert_to_unicode(words)
        if self.max_len:
            words = words[:self.max_len]
        for w in words:
            out_ids.append(self._transform2id(w))
        if padding and self.max_len:
            while len(out_ids) < self.max_len:
                out_ids.append(0)
        return out_ids
    
    def _transform_intent2ont_hot(self, words, padding=0):
        # 将多标签意图转为 one_hot
        out_ids = np.zeros(self.size, dtype=np.float32)
        words = convert_to_unicode(words)
        for w in words:
            out_ids[self._transform2id(w)] = 1.0
        return out_ids

    def _transform_seq2bert_id(self, words, padding=0):
        out_ids, seq_len = [], 0
        words = convert_to_unicode(words)
        if self.max_len:
            words = words[:self.max_len]
        seq_len = len(words)
        # 插入 [CLS], [SEP]
        out_ids.append(self._transform2id("[CLS]"))
        for w in words:
            out_ids.append(self._transform2id(w))
        mask_ids = [1 for _ in out_ids]
        if padding and self.max_len:
            while len(out_ids) < self.max_len + 1:
                out_ids.append(0)
                mask_ids.append(0)
        seg_ids = [0 for _ in out_ids]
        return out_ids, mask_ids, seg_ids, seq_len

    def transform(self, seq_list, is_bert=0):
        if is_bert:
            return [self._transform_seq2bert_id(seq) for seq in seq_list]
        else:
            return [self._transform_seq2id(seq) for seq in seq_list]

    def __len__(self):
        return len(self.voc2id)

class CatSLU(object):
    def __init__(self, dir_, ):
        self.dir_ = './data/catslu/catslu_traindev/data/map/'
        ontology = self.load_ontology(self.dir_ + "ontology.json")
        train_ = read_json(self.dir_ + 'train.json')
        dev_ = read_json(self.dir_ + 'development.json')
        # 判断哪些槽位没有在query中的，


    @staticmethod
    def load_ontology(ontology_file_path):
        src_base_dir = os.path.dirname(ontology_file_path)
        ontology = json.load(open(ontology_file_path))

        for slot in list(ontology['slots']['informable']):
            values = ontology['slots']['informable'][slot]
            ## load lexicon file
            if type(values) == str:
                values = [line.strip() for line in open(os.path.join(src_base_dir, values)) if line.strip() != ""]
            ontology['slots']['informable'][slot] = set(values)

        return ontology
    
    # 读取文件，./data/catslu/catslu_traindev/data/
    

if __name__ == "__main__":
    pass