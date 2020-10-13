#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/10/13 20:33:50
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
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

if __name__ == "__main__":
    pass