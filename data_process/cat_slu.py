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
        self.dir_ = './data/catslu/catslu_traindev/data/map/'
        ontology = self.load_ontology(self.dir_ + "ontology.json")
        train_ = read_json(self.dir_ + 'train.json')
        dev_ = read_json(self.dir_ + 'development.json')
        # 判断哪些槽位没有在query中的，
        train_arr = []
        for dlg in train_:
            for turn in  dlg['utterances']:
                q = turn['manual_transcript']
                sm = turn['semantic']
                # acts分类：inform， deny， request
                for slots in sm:
                    if len(slots) == 3:
                        act, k, v = slots
                    elif len(slots) == 2:
                        act, v = slots
                    if v not in q:
                        print(q)
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
    

if __name__ == "__main__":
    CatSLU()
    pass