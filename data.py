#!/usr/bin/env python
# encoding=utf-8
'''
Author: 	zhiyang.zzy 
Date: 		2020-09-23 22:46:15
Contact: 	zhiyangchou@gmail.com
FilePath: /ChineseNLU/data.py
Desc: 		catslu数据处理，作为数据输入格式。
todo: 
- 纯实体
- 
'''
# here put the import lib
from inspect import isbuiltin
import json
import random
from tensorflow.python.keras.backend import dtype
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import sequence
import math
import numpy as np
from util import read_file, convert_to_unicode

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
        # words = convert_to_unicode(words)
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
        if len(words) == 0:
            return out_ids
        for w in words:
            out_ids[self._transform2id(w)] = 1.0
        return out_ids

    def _transform_seq2bert_id(self, words, padding=0):
        out_ids, seq_len = [], 0
        # words = convert_to_unicode(words)
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

class OSDataset(object):
    """意图和实体数据集集"""
    def __init__(self, file_path, cfg, has_label=1, is_train=0):
        # 需要传入 ontology字典，将意图，槽位之类的做转换。
        self.file_path = file_path
        self.is_train = is_train
        self.cfg = cfg
        # word 配置
        self.word_vocab = Vocabulary(cfg['bert_dir'] + cfg['bert_vocab'], allow_unk=1, unk='[UNK]', pad='[PAD]', max_len=cfg['max_seq_len'])
        # intent 配置
        self.intent_vocab = Vocabulary(cfg['meta_dir'] + cfg['intent_file'])
        # tag 配置
        self.tag_vocab = Vocabulary(cfg['meta_dir'] + cfg['tag_file'], max_len=cfg['max_seq_len'])
        if has_label:
            self.get_label_dataset()
        else:
            self.get_unlabel_dataset()
        pass

    def get_label_dataset(self):
        # 读取数据集，然后将其转为对应格式
        self.dataset, self.id_set = [], []
        for line in read_file(self.file_path, "\1"):
            query, q_arr, tags, di, intents = line[:5]
            intents = intents.split("\3")
            q_arr = q_arr.split("\3")
            tags = tags.split("\3")
            self.dataset.append([query, q_arr, intents, tags])
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
            intents_ids = self.intent_vocab._transform_intent2ont_hot(intents)
            tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
            self.id_set.append([q_ids, mask_ids, seg_ids, seq_len, intents_ids, tags_ids])
            # self.id_set.append([q_ids, mask_ids, seg_ids, intents_ids, tags_ids, q_arr, intents, tags])
        pass
    
    def get_unlabel_dataset(self):
        # 读取数据集，这部分数据集只做预测
        self.dataset, self.id_set = [], []
        for line in read_file(self.file_path, "\1"):
            query = line[0]
            self.dataset.append([query])
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(query, padding=1)
            self.id_set.append([q_ids, mask_ids, seg_ids, seq_len])
        pass

    def get_batch(self, batch_size=None):
        if self.is_train:
            random.shuffle(self.id_set)
        if not batch_size:
            batch_size = self.cfg['batch_size']
        steps = int(math.ceil(float(len(self.id_set)) / batch_size))
        for i in range(steps):
            idx = i * batch_size
            cur_set = self.id_set[idx: idx + batch_size]
            yield zip(*cur_set)
        pass

    def __iter__(self):
        for each in self.id_set:
            yield each
    def __len__(self):
        return len(self.id_set)

class CatSLU(object):
    """catslu数据集，已经处理成intent+slot sequence格式"""
    def __init__(self, file_path, cfg, has_label=1, is_train=0):
        self.file_path = file_path
        self.is_train = is_train
        self.cfg = cfg
        # word 配置
        self.word_vocab = Vocabulary(cfg['bert_dir'] + cfg['bert_vocab'], allow_unk=1, unk='[UNK]', pad='[PAD]', max_len=cfg['max_seq_len'])
        # intent 配置
        self.intent_vocab = Vocabulary(cfg['meta_dir'] + cfg['intent_file'])
        # tag 配置
        self.tag_vocab = Vocabulary(cfg['meta_dir'] + cfg['tag_file'], max_len=cfg['max_seq_len'])
        if has_label:
            self.get_label_dataset()
        else:
            self.get_unlabel_dataset()
        pass
    def get_label_dataset(self):
        # 读取数据集，然后将其转为对应格式
        self.dataset, self.id_set = [], []
        for line in read_file(self.file_path, "\t"):
            query, intents, tags = line[:3]
            intents = intents.split("\3") if intents != "" else []
            q_arr = query
            tags = tags.split("\3")
            self.dataset.append([query, q_arr, intents, tags])
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
            intents_ids = self.intent_vocab._transform_intent2ont_hot(intents)
            tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
            self.id_set.append([q_ids, mask_ids, seg_ids, seq_len, intents_ids, tags_ids])
            # self.id_set.append([q_ids, mask_ids, seg_ids, intents_ids, tags_ids, q_arr, intents, tags])
            pass
        pass
    
    def get_unlabel_dataset(self):
        # 读取数据集，这部分数据集只做预测
        self.dataset, self.id_set = [], []
        for line in read_file(self.file_path, "\1"):
            query = line[0]
            self.dataset.append([query])
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(query, padding=1)
            self.id_set.append([q_ids, mask_ids, seg_ids, seq_len])
        pass

    def get_batch(self, batch_size=None):
        if self.is_train:
            random.shuffle(self.id_set)
        if not batch_size:
            batch_size = self.cfg['batch_size']
        steps = int(math.ceil(float(len(self.id_set)) / batch_size))
        for i in range(steps):
            idx = i * batch_size
            cur_set = self.id_set[idx: idx + batch_size]
            yield zip(*cur_set)
        pass

    def __iter__(self):
        for each in self.id_set:
            yield each
    def __len__(self):
        return len(self.id_set)

if __name__ == "__main__":
    # data_path = "data/catslu/catslu_traindev/data/map/train.json"
    # get_train_catslu_data(data_path)
    # file_dir = "/mnt/nlp/zhiyang.zzy/project/public/ChineseNLU/data/smp2020/SMP2020-ECDT_Origin_3shot/"
    # get_smp2020(file_dir)
    # data_dir = "/mnt/nlp/zhiyang.zzy/project/python3project/data_warehouse_code/data/out/"
    # date = "20200807"
    # train_file, dev_file, test_file = data_dir + "train." + date + ".seg", data_dir + "val." + date + ".seg", data_dir + "test." + date + ".seg", 
    cfg_path = "./config/config_catslu_map.yml"
    cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    train_set = CatSLU(cfg['data_dir'] + cfg['train_file'], cfg, has_label=1, is_train=1)
    for each in train_set.get_batch(4):
        batch = list(each)
        w_ids, mask_ids, seg_ids, seq_len, intents_ids, tags_ids = each
        # w_ids, mask_ids, seg_ids, seq_len, intents_ids, tags_ids = each
        # w_ids, mask_ids, seg_ids, seq_len, intents_ids, tags_ids, q_arr, intents, tags = each
        pass

