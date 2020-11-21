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

class CatSLU(object):
    def __init__(self, ):
        # 以下作为分类：
        # request所有
        # map域：
        # 'inform', '操作; 'value', 'dontcare'; 终点名称, 公司; 
        # 操作 大部分都在，但是 退出 等不在
        # 请求类型: 几个附近、周边、旁边都是相似的。定位是不一样的。（定位也在inform-对象中）
        # 如果是其他槽位，如果不在query中，那么case略过。
        # music: '歌曲名', 'dontcare';
        # 迁移的域，样本较少，考虑其他域呢

        self.dir_ = './data/catslu/catslu_traindev/data/music/'
        ontology = self.load_ontology(self.dir_ + "ontology.json")
        train_ = read_json(self.dir_ + 'train.json')
        dev_ = read_json(self.dir_ + 'development.json')
        test_ = read_json('./data/catslu/catslu_test/data/map/test.json')
        # 判断哪些槽位没有在query中的，
        train_arr = []
        for dlg in train_:
            for turn in  dlg['utterances']:
                q = turn['manual_transcript']
                if 'unknown' in q:
                    continue
                sm = turn['semantic']
                # acts分类：inform， deny， request
                for slots in sm:
                    if len(slots) == 3:
                        act, k, v = slots
                    elif len(slots) == 2:
                        act, v = slots
                    # if k in ['请求类型', '对象', 'value'] or act == 'request':
                    #     continue
                    if v not in q:
                        print(q, slots)
                        # print()
        pass


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
    
class SMP2019(object):
    def __init__(self, file_):
        self.file = file_
        pass
    pass

def split_smp(file_):
    with open(file_) as f:
        data_arr = json.load(f)
    # 随机切分为训练集，验证集，测试集。
    pass

if __name__ == "__main__":
    CatSLU()
    pass