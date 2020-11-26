#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/11/26 15:23:01
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   预处理
1. 修正asr错误，通过查看slot value是否在ontology中，判断asr是否错误，如果asr错误，那么对slot value进行修正。算是track。
2. 将catslu的三元组格式处理成intent、slots格式。

'''

# here put the import lib
from collections import defaultdict
import os
from util import convert_to_unicode, read_file, read_json, contain_eng
import numpy as np
import json

def load_ontology(ontology_file_=""):
    src_base_dir = os.path.dirname(ontology_file_)
    ontology = read_json(ontology_file_)

    for slot in list(ontology['slots']['informable']):
        values = ontology['slots']['informable'][slot]
        ## load lexicon file
        if type(values) == str:
            values = [line.strip() for line in open(os.path.join(src_base_dir, values)) if line.strip() != ""]
        ontology['slots']['informable'][slot] = set(values)

    return ontology

def catslu_process_map(file_train, file_dev=None, file_test=None):
    """
    原始文件是json
    """
    # 将test 和 train 合并。
    train_ = read_json(file_train)
    if file_dev:
        dev_ = read_json(file_dev)
        train_ += dev_
    if file_test:
        test_ = read_json(file_test)
        train_ += test_
    # 统计 act-slot 有哪些？
    cnt, not_v_cnt = 0, 0
    unk_len, act_slots, act_slot_value = [], defaultdict(int), defaultdict(int)
    kv_map = defaultdict(int)
    for dig in train_:
        for turn in dig['utterances']:
            # 包含 "(" 且 semantic 为空的，跳过。
            q = turn['manual_transcript']
            if "(" in q and len(turn['semantic']) == 0:
                continue
            is_contain_unk = 1 if "(" in q else 0
            # q = q.replace("(unknown)", ',').replace("(side)", ',').replace('(dialect)', ',').replace('(robot)', ',')
            q = q.replace("(unknown)", '').replace("(side)", ',').replace('(dialect)', '').replace('(robot)', ',')
            if q == "":
                print(q, turn)
                continue
            if is_contain_unk:
                unk_len.append(len(q))
                cnt += 1
            # if is_contain_unk and len(q) > 25:
            #     # print(q, turn)
            #     continue
            for slot in turn['semantic']:
                if len(slot) == 3:
                    act, k, v = slot
                    act_slot_value["-".join([act, k])] += 1
                    kv = "-".join([act, k, v])
                    if v not in q:
                        if kv not in ['inform-操作-退出','inform-value-dontcare','inform-请求类型-周边','inform-请求类型-最近','inform-操作-重新导航','inform-操作-继续导航','inform-终点名称-公司']:
                            print("v not in q:", q, act, k, v)
                            not_v_cnt += 1
                        else: 
                            kv_map[kv] += 1
                elif len(slot) == 2:
                    act, k = slot
                    act_slots["-".join(slot)] += 1
                else:
                    print("worng slot:", slot)
                pass
    print(act_slots)
    print(act_slot_value)
    print(kv_map)
    print(cnt, not_v_cnt, max(unk_len), )
    # act-slot-value：中有哪些act-slot
    # act-slot-value：中有哪些value不在query中
    # 注意semantic为空query不为空的句子，怎么处理
    

    pass

if __name__ == "__main__":
    domain = 'map'
    data_dir = './data/catslu/catslu_traindev/data/{}/'.format(domain)
    test_file = './data/catslu/catslu_test/data/{}/test.json'.format(domain)
    catslu_process_map(data_dir + "train.json", data_dir + "development.json", test_file)
    pass