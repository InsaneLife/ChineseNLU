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
import json


def get_train_data(file_):
    print("test")
    with open(file_) as f:
        data_set = json.load(f)
    
    pass

if __name__ == "__main__":
    data_path = "data/catslu/catslu_traindev/data/map/train.json"
    get_train_data(data_path)
    pass