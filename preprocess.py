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
from util import convert_to_unicode, read_file, read_json, contain_eng, write_file
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

def catslu_process_map(file_train, file_dev=None, file_test=None, cur_kv=None):
    """
    原始文件是json，将map域数据转为nlu输入格式，[query, category, slots]
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
                        if kv not in cur_kv:
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
    for each in act_slots.keys():
        print(each)
    for each in cur_kv:
        print(each)
    print("-" * 50)
    print(act_slot_value)
    for each in act_slot_value.keys():
        print("B_" + each)
        print("I_" + each)
    print("-" * 50)
    print(kv_map)
    print(cnt, not_v_cnt, max(unk_len), )
    # act-slot-value：中有哪些act-slot
    # act-slot-value：中有哪些value不在query中
    # 注意semantic为空query不为空的句子，怎么处理
    pass

def match_index(norm_query_sep, v):
    '''
    norm_query_sep:查询词归一化之后按字拆分，我 想 去 北 京 天 安 门
    v:slots归一化之后按字拆分 北 京 天 安 门
    return: start_index(v在norm_query_sep出现的起始index), end_index（v在norm_query_sep出现的结束index），从0开始
    返回slots的所有匹配可能
    '''
    start_index = -1
    end_index = -1
    flag = 0
    i = 0
    j = 0
    i0 = 0
    start_end = []
    while i < len(norm_query_sep):
        p = norm_query_sep[i]
        q = v[j]
        if p == q:
            if flag == 0:
                start_index = i
            flag = 1
            i += 1
            j += 1
            if j == len(v):
                start_end.append([start_index, start_index + len(v) - 1])
                flag = 0
                j = 0
                i0 += 1
                i = i0
        else:
            i0 += 1
            i = i0
            j = 0
            flag = 0
    if flag == 1 and len(norm_query_sep) - start_index == len(v):
        return [start_index, start_index + len(v) - 1]
    if len(start_end) == 0:
        start_end.append([-1, -1])
    return start_end

def unifySlots(unify_start_end):
    # 对槽位的起始和终止位置进行梳理，比如：放下最新歌榜的歌 type:新歌榜|category:歌
    # 新歌榜:[3,5] 歌:[[4,4],[7,7]]
    #    slots_index = unifySlots(unify_start_end)
    # slots_index = 新歌榜:[3,5] 歌:[7,7] 我想听张国荣的我
    new_start_end = {}
    if len(unify_start_end) == 1:
        for key in unify_start_end:
            new_start_end[key] = unify_start_end[key][0]
        return new_start_end
    sodic = sorted(unify_start_end.items(), key=lambda asd: asd[1][0][0], reverse=False)
    #    for key in sodic:
    #        print key[0]
    #        print key[1]
    #    print "========"
    before_word = []
    after_word = []
    for idx in range(len(sodic)):
        word = sodic[idx]
        if len(word[1]) == 1:
            new_start_end[word[0]] = word[1][0]
    for idx in range(len(sodic)):
        word = sodic[idx]
        if word[0] in new_start_end:
            continue
        # 仅对有多对start end的词进行处理
        if idx == 0:
            after_word_key = sodic[idx + 1][0]
            if after_word_key not in new_start_end:  # 如果后面的key也是多个start_end，则选择第一个
                new_start_end[word[0]] = word[1][0]
            else:
                for idx_range in word[1]:
                    s, e = idx_range
                    after_s, after_e = sodic[idx + 1][1][0]
                    if (e < after_s) or (s > after_e):
                        new_start_end[word[0]] = idx_range
                        break
        elif idx + 1 < len(sodic):
            after_word_key = sodic[idx + 1][0]
            if after_word_key not in new_start_end:  # 如果后面的key也是多个start_end，则选择第一个
                new_start_end[word[0]] = word[1][0]
                continue
            for idx_range in word[1]:
                s, e = idx_range
                before_word = new_start_end[sodic[idx - 1][0]]
                before_s, before_e = before_word
                after_s, after_e = sodic[idx + 1][1][0]
                if (e < after_s and s > before_e) or (s > after_e):
                    new_start_end[word[0]] = idx_range
                    break
        else:
            before_word = new_start_end[sodic[idx - 1][0]]
            before_s, before_e = before_word
            for idx_range in word[1]:
                s, e = idx_range
                if s > before_e:
                    new_start_end[word[0]] = idx_range
                    break
    return new_start_end

def setTagWord(query:str, slots:list):
    '''
    norm_qeury=我想去北京天安门
    slots_list=[dst, 北京天安门]
    return:我:O 想:O 去:O 北:B_DST 京:I_DST 天:I_DST 安:I_DST 门:I_DST
    '''
    # 如果槽位为空
    if len(slots) == 0:
        return '\3'.join(['O' for _ in query])
    unify_start_end, mydic = {}, []
    for slot in slots:
        if len(slot) != 2:
            print("worng slot:", slot)
            continue
        k, v = slot
        start_end = match_index(query, v)
        unify_start_end[k + "\1" + v] = start_end
    slots_index = unifySlots(unify_start_end)
    if len(slots_index) != len(unify_start_end):
        print('[ERROR] query=%s' % query)
        print(slots)
    for slots in slots_index:
        #            print 'start_index=%d'%start_index
        #            print 'end_index=%d'%end_index
        start_index, end_index = slots_index[slots]
        if start_index == -1 and end_index == -1:  # 没找到slots在query中出现的位置
            continue
        k, uu = slots.split('\1')
        mydic.append([start_index, end_index, k, uu])
    mydic.sort(key=lambda x:x[0], reverse=False)
    index = 0
    r = []
    for s in range(len(query)):
        #            print 's=%s'%s
        flag = 0
        for si, ei, k, _ in mydic:
            if s >= si and s <= ei:
                if s == si:
                    r.append([query[s], 'B_' + k])
                else:
                    r.append([query[s], 'I_' + k])
                flag = 1
                break
        if flag == 0:
            r.append([query[s], 'O'])
    return '\3'.join([m[1] for m in r])

def catslu_map(file_, cate_kv):
    ''' 输入原始json，输出格式：[{query, intents, seq_label, slots}]
    字母小写化
    '''
    data_arr = read_json(file_)
    out_arr = []
    for dig in data_arr:
        for turn in dig['utterances']:
            q = turn['manual_transcript']
            if "(" in q and len(turn['semantic']) == 0:
                continue
            is_contain_unk = 1 if "(" in q else 0
            # q = q.replace("(unknown)", ',').replace("(side)", ',').replace('(dialect)', ',').replace('(robot)', ',')
            q = q.replace("(unknown)", '').replace("(side)", ',').replace('(dialect)', '').replace('(robot)', ',')
            if q == "":
                print(q, turn)
                continue
            slots = []
            intents = []
            for slot in turn['semantic']:
                if len(slot) == 2:
                    intents.append("-".join(slot))
                if len(slot) == 3:
                    act, k, v = slot
                    if "-".join(slot) in cate_kv:
                        intents.append("-".join(slot))
                    else:
                        if v not in q:
                            print("v not in q:", q, slot)
                        slots.append([act + "-" + k, v])
            tag_seq = setTagWord(q, slots)
            slots_str = "|".join([":".join(x) for x in slots])
            out_arr.append([q, '\3'.join(intents), tag_seq, slots_str])
    write_file(out_arr, file_ + ".seg")

if __name__ == "__main__":
    # s = "播放she"
    # print(contain_eng(s))
    mp_kv = ['inform-操作-退出','inform-value-dontcare','inform-请求类型-周边','inform-请求类型-最近','inform-操作-重新导航','inform-操作-继续导航','inform-终点名称-公司']
    music_kv = ['inform-value-dontcare', 'inform-歌曲名-dontcare']
    video_kv = ['inform-value-dontcare']
    weather_kv = ["inform-气象-日落"]
    kv_map = {'map': mp_kv, 'music': music_kv, 'video': video_kv, 'weather': weather_kv}
    domain = 'weather'
    cur_kv = kv_map[domain]
    data_dir = './data/catslu/catslu_traindev/data/{}/'.format(domain)
    test_file = './data/catslu/catslu_test/data/{}/test.json'.format(domain)
    catslu_process_map(data_dir + "train.json", data_dir + "development.json", test_file, cur_kv)

    # seq, tag_seq = setTagWord(query='歌放下最新歌榜的歌', slots=[["type", '新歌榜'], ['category', '歌']])
    cate_kv = ['inform-操作-退出','inform-value-dontcare','inform-请求类型-周边','inform-请求类型-最近','inform-操作-重新导航','inform-操作-继续导航','inform-终点名称-公司']
    domain = 'map'
    print(domain)
    data_dir = './data/catslu/catslu_traindev/data/{}/'.format(domain)
    test_file = './data/catslu/catslu_test/data/{}/test.json'.format(domain)
    catslu_map(data_dir + "train.json", cate_kv)
    catslu_map(data_dir + "development.json", cate_kv)
    catslu_map(test_file, cate_kv)

    cate_kv = ['inform-value-dontcare', 'inform-歌曲名-dontcare']
    domain = 'music'
    print(domain)
    data_dir = './data/catslu/catslu_traindev/data/{}/'.format(domain)
    test_file = './data/catslu/catslu_test/data/{}/test.json'.format(domain)
    catslu_map(data_dir + "train.json", cate_kv)
    catslu_map(data_dir + "development.json", cate_kv)
    catslu_map(test_file, cate_kv)
    
    cate_kv = ['inform-value-dontcare']
    domain = 'video'
    print(domain)
    data_dir = './data/catslu/catslu_traindev/data/{}/'.format(domain)
    test_file = './data/catslu/catslu_test/data/{}/test.json'.format(domain)
    catslu_map(data_dir + "train.json", cate_kv)
    catslu_map(data_dir + "development.json", cate_kv)
    catslu_map(test_file, cate_kv)

    cate_kv = ["inform-气象-日落"]
    domain = 'weather'
    print(domain)
    data_dir = './data/catslu/catslu_traindev/data/{}/'.format(domain)
    test_file = './data/catslu/catslu_test/data/{}/test.json'.format(domain)
    catslu_map(data_dir + "train.json", cate_kv)
    catslu_map(data_dir + "development.json", cate_kv)
    catslu_map(test_file, cate_kv)
    pass