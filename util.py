#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/10/13 20:33:50
@Author  :   Zhiyang.zzy 
@Contact :   Zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
import os 
import six
import time
import json
import re

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def read_file(file_:str, splitter:str=None):
    out_arr = []
    with open(file_, encoding="utf-8") as f: 
        out_arr = [x.strip("\n") for x in f.readlines()]
        if splitter:
            out_arr = [x.split(splitter) for x in out_arr]
    return out_arr

def read_json(file_:str):
    out_arr = []
    with open(file_, encoding="utf-8") as f: 
        out_arr = json.load(f)
    return out_arr

def write_json(out_json, file_:str):
    with open(file_, 'w', encoding="utf-8") as f: 
        json.dump(out_json, f, ensure_ascii=False)

def get_chunk_type(tok, id2tag):
    """
    :param tok:     tok = 4
    :param id2tag:  {4: "B_keyword", "3": "I_keyword", ...}
    :return:
    """
    tag_name = id2tag[tok]
    tag_info = tag_name.split("_", 1)
    tag_class, tag_type = tag_info[0], tag_info[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    :param seq:  seq = [ 4 5 0 3]
    :param tags: tags = data_parser.tag_idx, {"B_type": 0, "I_type": 1, ...}
    :return:     result = [("keyword", 0, 2), ("type", 3, 4), ...]
    """
    default = []
    if NONE in tags:
        default.append(tags[NONE])
    if "o" in tags:
        default.append(tags[NONE])
    id2tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        if tok in default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        elif tok not in default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, id2tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def chunks_to_slot(query, chunks):
    """
    :param query: 查询串
    :param chunks: [("keyword", 0, 2), ("type", 3, 4)]
    :return: [["keyword", "***"], ["type", "***"]]
    """
    query = convert_to_unicode(query)
    slots = []
    for chunk in chunks:
        if len(chunk) != 3:
            continue
        slot_type, start_pos, end_pos = chunk[:]
        slots.append([slot_type, query[start_pos:end_pos]])
    return slots

def alignment(sentence, tags):
    """
    sentence: 一句话
    tags: sentence中每个word对应的tag组成的列表
    """
    sentence = convert_to_unicode(sentence)
    entity_type = []
    if len(sentence) != len(tags):
        print("Not Matched: {} <==> {}".format(sentence, " ".join(tags)))
        return entity_type
    pos_start, pos_end = 0, 0
    cur_tag = ""
    # entity_type = {}

    for idx, w in enumerate(sentence):
        if tags[idx].startswith("B_"):
            if cur_tag:
                pos_end = idx - 1
                # entity_type[sentence[pos_start:pos_end+1].encode("utf-8")] = cur_tag
                entity_type.append((sentence[pos_start:pos_end + 1].encode("utf-8"), cur_tag))
            cur_tag = '_'.join(tags[idx].split("_")[1:])
            pos_start = idx
        elif not tags[idx].startswith("I_"):
            if cur_tag:
                pos_end = idx - 1
                # entity_type[sentence[pos_start:pos_end+1].encode("utf-8")] = cur_tag
                entity_type.append((sentence[pos_start:pos_end + 1].encode("utf-8"), cur_tag))
            cur_tag = ""
        else:
            if cur_tag != '_'.join(tags[idx].split("_")[1:]):
                print("Tag Not Matched of B_ and I_. {} <==> {}".format(sentence, " ".join(tags)))
    if cur_tag:
        # entity_type[sentence[pos_start:].encode("utf-8")] = cur_tag
        entity_type.append((sentence[pos_start:].encode("utf-8"), cur_tag))
    return entity_type

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

def contain_eng(str0):
    return bool(re.search('[a-z]', str0))

if __name__ == "__main__":
    pass