import re
import os
import sys
import time
import json
import nltk
import torch
import random
import argparse
import operator
import progressbar
import numpy as np
from torch import nn
import torch.nn.functional as F
from operator import itemgetter

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

def e2e_batch_generate(model, one_inference_batch, input_contain_db, data):
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
    bs_batch, da_batch, nlg_batch, parse_dict_batch = one_inference_batch
    batch_size = len(parse_dict_batch)
    res_batch_parse_dict = parse_dict_batch

    # if input_contain_db == True: then we first generate the belief state and get the db result
    # otherwise, we perform the generation all in-parallel

    bs_tensor, bs_mask = data.pad_batch(bs_batch)
    if is_cuda:
        bs_tensor = bs_tensor.cuda(device)
        bs_mask = bs_mask.cuda(device)

    batch_bs_text = model.batch_generate(bs_tensor, bs_mask, generate_mode='bs', max_decode_len=max_response_len)

    # the belief state sequence could be long
    batch_bs_restore_text = []
    for idx in range(batch_size):
        one_bs_text = batch_bs_text[idx]
        res_batch_parse_dict[idx]['bspn_gen'] = one_bs_text

    if input_contain_db:
        # we need to query the db base
        batch_db_input_id_list = []
        for idx in range(batch_size):
            one_queried_db_result = \
            data.reader.bspan_to_DBpointer(batch_bs_text[idx], res_batch_parse_dict[idx]['turn_domain'])
            one_db_text = '<sos_db> ' + one_queried_db_result + ' <eos_db>' 
            one_db_token_id_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_db_text))
            batch_db_input_id_list.append(one_db_token_id_input)
    else:
        batch_db_input_id_list = []
        for _ in range(batch_size):
            batch_db_input_id_list.append([])

    if input_contain_db:
        # then we generate the dialogue action
        da_batch_input_id_list = []
        for idx in range(batch_size):
            one_da_input_id_list = da_batch[idx] + batch_db_input_id_list[idx]
            da_batch_input_id_list.append(one_da_input_id_list)
        da_tensor, da_mask = data.pad_batch(da_batch_input_id_list)
    else:
        da_tensor, da_mask = data.pad_batch(da_batch)

    if is_cuda:
        da_tensor = da_tensor.cuda(device)
        da_mask = da_mask.cuda(device)
    batch_da_text = model.batch_generate(da_tensor, da_mask, generate_mode='da', max_decode_len=max_span_len)


    for idx in range(batch_size):
        res_batch_parse_dict[idx]['aspn_gen'] = batch_da_text[idx]   

    if input_contain_db:
        # finally, we generate the response
        nlg_batch_input_id_list = []
        for idx in range(batch_size):
            one_nlg_input_id_list = nlg_batch[idx] + batch_db_input_id_list[idx]
            nlg_batch_input_id_list.append(one_nlg_input_id_list)
        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch_input_id_list)
    else:
        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch)

    if is_cuda:
        nlg_tensor = nlg_tensor.cuda(device)
        nlg_mask = nlg_mask.cuda(device)
    batch_nlg_text = model.batch_generate(nlg_tensor, nlg_mask, generate_mode='nlg', max_decode_len=max_response_len)
    for idx in range(batch_size):
        res_batch_parse_dict[idx]['resp_gen'] = batch_nlg_text[idx]
    return res_batch_parse_dict
