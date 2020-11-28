#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/10/19 15:35:11
@Author  :   Zhiyang.zzy 
@Desc    :   
'''

# here put the import lib
import yaml
from data import CatSLU
from model.joint_bert import JointBert
import os 
import logging
import argparse
logging.basicConfig(level=logging.INFO)

def join_bert_train(cfg_path = "./config/catslu_map.yml"):
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    train_set = CatSLU(cfg['data_dir'] + cfg['dev_file'], cfg, is_train=1)
    dev_set, test_set = train_set, train_set
    # train_set = CatSLU(cfg['data_dir'] + cfg['train_file'], cfg, is_train=1)
    # dev_set = CatSLU(cfg['data_dir'] + cfg['dev_file'], cfg)
    # test_set = CatSLU(cfg['test_file'], cfg)
    print("train size:{};dev size:{};test size:{};".format(len(train_set), len(dev_set), len(test_set)))
    model = JointBert(cfg, train_set.intent_vocab, train_set.tag_vocab)
    model.fit(train_set, dev_set, test_set)

def join_bert_predict(use_crf=1, predict_file="./result/osnlu/unk_open"):
    cfg_path = "./config/config.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    if use_crf:
        cfg['use_crf'], postfix = 1, ".crf"
        cfg['checkpoint_dir'] = "result/checkpoint/bert_1018_crf/model"
    else:
        cfg['use_crf'], postfix = 0, ""
        cfg['checkpoint_dir'] = "result/checkpoint/bert_1019/model"
    predict_set = CatSLU(predict_file, cfg, has_label=0)
    print("test size:{};".format(len(predict_set)))
    model = JointBert(cfg, predict_set.intent_vocab, predict_set.tag_vocab)
    # 加载模型
    model.restore_session(cfg['checkpoint_dir'])
    # 预测
    result = model.predict_sf(predict_set)
    with open(predict_file + ".pipe" + postfix, 'w', encoding='utf-8') as out:
        for line in result:
            out.write(line + "\n")
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", default="./config/catslu_video.yml", type=str, help="input file of unaugmented data")
    ap.add_argument("--device", default="5", type=str, help="CUDA_VISIBLE_DEVICES")
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    join_bert_train(args.cfg_path)
    # predict_file="./result/osnlu/unk_open"
    # print(predict_file)
    # join_bert_predict(use_crf=1, predict_file=predict_file)
    pass