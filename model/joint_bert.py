#!/usr/bin/env python
# encoding=utf-8
'''
Author: 	Zhiyang.zzy 
Date: 		2020-10-18 23:57:33
Contact: 	Zhiyangchou@gmail.com
FilePath: /ChineseNLU/joint_bert.py
Desc: 		
'''
# here put the import lib
import logging
import tensorflow as tf
import math
import numpy as np
from .base_model import BaseModel
from util import get_chunks, chunks_to_slot
from .util import computeF1Score

class JointBert(BaseModel):
    def __init__(self, cfg, intent_vocab, tag_vocab):
        self.cfg = cfg
        self.intent_vocab = intent_vocab
        self.intent_num = intent_vocab.size
        self.tag_vocab = tag_vocab
        self.tag_num = tag_vocab.size
        self.build()
        pass

    def _forward(self):
        # 添加bert
        self.add_bert_layer(use_bert_pre=1)
        # with tf.variable_scope("dropout"):
        #     _query_output = self._dropout(self.bert_output_seq)
        #     output_classify = self._dropout(self.cls_output)
        # _query_output = self.bert_output_seq
        # self.cls_output
        # nsteps = tf.shape(self.bert_output_seq)[1]

        self.logit_intent = tf.layers.dense(self.cls_output, self.intent_num)
        self.logit_tagger = tf.layers.dense(self.bert_output_seq, self.tag_num)
        # self.tag_logit = tf.reshape(
        #     self.tag_logit, [-1, nsteps, self.bert_config.hidden_size])
        pass

    def _add_loss(self):
        with tf.variable_scope("intent_loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_intent, labels=self.intents)
            self.loss_intent = tf.reduce_mean(losses)
        with tf.variable_scope("slot_loss"):
            # todo: 添加crf
            if self.cfg['use_crf']:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logit_tagger,
                                                                                 self.tags,
                                                                                 self.query_length)
                self.trans_params = trans_params
                self.loss_tagger = tf.reduce_mean(-log_likelihood)
                # self.loss_tagger = 0
            else:
                labels_one_hot = tf.one_hot(self.tags, self.tag_num)
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_tagger, labels=labels_one_hot, dim=-1)
                self.loss_tagger = tf.reduce_mean(losses)
        self.loss = self.loss_tagger + self.loss_intent
        tf.summary.scalar("intent_loss", self.loss_intent)
        tf.summary.scalar("slot_loss", self.loss_tagger)
        tf.summary.scalar("loss", self.loss)
    def _add_pred_op(self):
        with tf.variable_scope("predict"):
            if self.cfg['use_crf']:
                self.tag_pred, self.tag_pred_score = tf.contrib.crf.crf_decode(self.logit_tagger,
                                                                               self.trans_params,
                                                                               self.query_length)
            else:
                self.tag_pred = tf.argmax(self.logit_tagger, -1, output_type=tf.int64)
            self.intent_prob = tf.sigmoid(self.logit_intent)
            self.intent_pred = tf.argmax(self.logit_intent, -1, output_type=tf.int32)
            
    def build(self):
        self._add_placeholder()
        self._forward()
        self._add_loss()
        self._add_pred_op()
        self.add_train_op(self.cfg['optimizer'], self.cfg['bert_lr'], self.loss)
        self._init_session()
        self._add_summary()

    def _add_placeholder(self):
        self.query_ids = tf.placeholder(dtype=tf.int32,
                                        shape=[None, self.cfg['max_seq_len'] + 1],
                                        name="query_id")
        self.mask_ids = tf.placeholder(dtype=tf.int32,
                                       shape=[None, self.cfg['max_seq_len'] + 1],
                                       name="mask_id")
        self.seg_ids = tf.placeholder(dtype=tf.int32,
                                      shape=[None, self.cfg['max_seq_len'] + 1],
                                      name="seg_id")
        self.query_length = tf.placeholder(dtype=tf.int32,
                                           shape=[None],
                                           name="query_length")
        # label
        self.tags = tf.placeholder(dtype=tf.int32,
                                   shape=[None, None],
                                   name="tags")
        self.intents = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.intent_num],
                                      name="intents")
        # config
        self.is_train_place = tf.placeholder(
            dtype=tf.bool, name='is_train_place')
    def run_epoch(self, epoch, train, dev, test=None):
        nbatches = int(math.ceil(float(len(train)) / self.cfg['batch_size']))
        progbar = tf.keras.utils.Progbar(nbatches)
        for i, batch in enumerate(train.get_batch()):
            fd = self.build_train_fd(batch)
            # s = self.sess.run([self.embedding_table], feed_dict=fd)
            run_list = [self.train_op, self.loss, self.loss_intent, self.loss_tagger, self.merged]
            result = self.sess.run(run_list, feed_dict=fd)
            progbar.update(i+1, [('loss', result[1]), ('loss_intent', result[2]), ('loss_tagger', result[3])])
            if i % 1000 == 0:
                self.file_writer.add_summary(result[4], epoch * nbatches + i)
        eval_sf_acc = self.evaluate(dev)
        print("dev sf add:", eval_sf_acc)
        return eval_sf_acc

    def build_train_fd(self, batch):
        # batch = [_ids, mask_ids, seg_ids, mintent_ids, tags_ids]
        batch = list(batch)
        fd = {
            self.query_ids: batch[0],
            self.mask_ids: batch[1],
            self.seg_ids: batch[2],
            self.query_length: batch[3],
            self.intents: batch[4],
            self.tags: batch[5],
            self.is_train_place: True
        }
        return fd
    
    def predict(self, test):
        out_tags, out_intent_prob = [], []
        # 无标签数据
        for batch in test.get_batch():
            batch = list(batch)
            fd = {
                self.query_ids: batch[0],
                self.mask_ids: batch[1],
                self.seg_ids: batch[2],
                self.query_length: batch[3],
                self.is_train_place: False
            }
            # s = self.sess.run(self.embedding_table, feed_dict=fd)
            tag_pred, intent_prob = self.sess.run([self.tag_pred, self.intent_prob], feed_dict=fd)
            # tag_pred = self.sess.run(self.tag_pred, feed_dict=fd)
            # intent_prob = self.sess.run(self.intent_prob, feed_dict=fd)
            out_tags.extend(tag_pred)
            out_intent_prob.extend(intent_prob)
        return out_intent_prob, out_tags

    def predict_sf(self, test):
        # 预测意图和tag
        out_sf = []
        out_intent_prob, out_tags = self.predict(test)
        # 将意图id转为多个意图
        for pred_i, pred_t, batch, batch_str in zip(out_intent_prob, out_tags, test.id_set, test.dataset):
            pred_i = filter(lambda x: x > -1, [i if x >= 0.5 else -1 for i, x in enumerate(pred_i)])
            # 转换为string
            pred_i_str= [self.intent_vocab.id2voc[x] for x in pred_i]
            # tag 转字符串
            seq_len, query = batch[3], batch_str[0]
            pred_t = pred_t[:seq_len]
            tag_pred_chunks = set(get_chunks(pred_t, self.tag_vocab.voc2id))
            tag_pred_chunks = chunks_to_slot(query, tag_pred_chunks)
            slots_str = ";".join([":".join([k, v]) for k, v in tag_pred_chunks])
            out_sf.append("\t".join([query, "|".join(pred_i_str), slots_str]))
        return out_sf

    def evaluate(self, dev):
        """ 评估 sentence acc, intent acc, slot f1 """
        # intent可能是多标签
        intent_acc, tag_acc = [], []
        pred_intent_prob, pred_tags = self.predict(dev)
        true_tags_str, pred_tags_str = [], []
        # 评估对比
        for batch, pred_ip, pred_t in zip(dev.id_set, pred_intent_prob, pred_tags):
            seq_len, true_mintent_ids, true_tags_ids = batch[3:]
            # intent > 0.5 的 变成 1
            pred_ip = [1 if x >= 0.5 else 0 for x in pred_ip]
            s = list(true_mintent_ids) == pred_ip
            intent_acc.append(s)
            true_tags_ids, pred_t = true_tags_ids[:seq_len], pred_t[:seq_len]
            tag_acc.append(list(true_tags_ids) == list(pred_t))
            # 计算f1
            # true_chunks = set(get_chunks(tags_ids, self.tag_vocab.voc2id))
            # pred_chunks = set(get_chunks(pred_t, self.tag_vocab.voc2id))
            true_tags_str.append([self.tag_vocab.id2voc[x] for x in true_tags_ids])
            pred_tags_str.append([self.tag_vocab.id2voc[x] for x in pred_t])
            pass
        sf_acc = [a & b for a, b in zip(intent_acc, tag_acc)]
        sf_acc = np.mean(sf_acc)
        joint_metrics = {"accuracy": sf_acc}
        f1, precision, recall = computeF1Score(true_tags_str, pred_tags_str)
        print("slot f1:{}, precision:{}; recall:{}".format(f1, precision, recall))
        return sf_acc