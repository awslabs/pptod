import torch
import torch.nn as nn

import os
import random
import time
import json
import numpy as np
import nltk
import os
import sys
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import operator
from operator import itemgetter
import progressbar
import argparse

import re
def restore_text(text, mode):
    if mode == 'bs':
        text = re.sub(' is ', ' ', text)
    elif mode == 'da':
        pass
    else:
        raise Exception('Wrong Restore Mode!!!')
    text = re.sub(' , ', ' ', text)
    text = ' '.join(text.split()).strip()
    return text

def erase_error(text):
    # [value - -> [value-
    # [ value -> [value
    # [value- -> [value_
    text = re.sub(r"\[value -", r"[value\-", text)
    text = re.sub(r"\[ value", r"[value", text)
    text = re.sub(r"\[value-", r"[value_", text)
    return text

def batch_generate(model, one_inference_batch, data):
    is_cuda = next(model.parameters()).is_cuda
    if is_cuda: 
        #device = next(model.parameters()).device
        device = torch.device('cuda')
        if torch.cuda.device_count() > 1: # multi-gpu training 
            model = model.module
        else: # single gpu training
            pass
    else:
        device = 0

    max_span_len, max_response_len = 80, 120
    tokenizer = data.tokenizer
    bs_batch, parse_dict_batch = one_inference_batch
    batch_size = len(parse_dict_batch)
    res_batch_parse_dict = parse_dict_batch

    bs_tensor, bs_mask = data.pad_batch(bs_batch)
    if is_cuda:
        bs_tensor = bs_tensor.cuda(device)
        bs_mask = bs_mask.cuda(device)
    batch_bs_text = model.batch_generate(bs_tensor, bs_mask, generate_mode='bs', max_decode_len=max_response_len)
    for idx in range(batch_size):
        one_bs_text = batch_bs_text[idx]
        res_batch_parse_dict[idx]['bspn_gen'] = one_bs_text
    return res_batch_parse_dict
