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

if __name__ == "__main__":
    pass