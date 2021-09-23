import os
import re
import sys
import torch
import random
import argparse
import operator
import progressbar
import numpy as np
import torch.nn as nn
from operator import itemgetter
import torch.nn.functional as F

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

def batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db, input_contain_db, data):
    '''
        e2e evaluation: 
            ref_bs: False
            ref_act: False
            ref_db: False
            input_contain_db: True or False
            
            ************************************************************************************************
            In this case: bspn_gen, aspn_gen, resp_gen need to be generated, if input_with_db then the db 
                          should be queried from the database using the generated belief state
            ************************************************************************************************
            
        oracle evaluation:
            ref_bs: True
            ref_act: True
            ref_db: True
            input_contain_db: True or False
            
            ************************************************************************************************
            In this case: only the resp_gen need to be generated, if input_with_db then using the oracle db 
                          as input
            ************************************************************************************************
                          
        policy evaluation:
            ref_bs: True
            ref_act: False
            ref_db: True
            input_contain_db: True or False
            
            ************************************************************************************************
            In this case: the aspn and resp_gen need to be generated, if input_with_db then using the oracle db 
                          as input
            ************************************************************************************************

        This function deals with batch generation. In order to fully take advantage of batch inference,
        in each batch, we only generate one type of output. e.g. Given a batch of dialogue history, we 
        generate the corresponding belief state/dialogue action/system response for the given batch. The 
        specific type of output is decided by the input argument "generate_mode"
    '''

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

    reform_bs_and_act = False

    max_span_len, max_response_len = 80, 120
    tokenizer = data.tokenizer
    bs_batch, da_batch, nlg_batch, parse_dict_batch = one_inference_batch
    batch_size = len(parse_dict_batch)
    res_batch_parse_dict = parse_dict_batch

    if ref_bs == False and ref_act == False and ref_db == False:
        evaluation_setup = 'e2e'
    elif ref_bs == True and ref_act == True and ref_db == True:
        evaluation_setup = 'oracle'
    elif ref_bs == True and ref_act == False and ref_db == True:
        evaluation_setup = 'policy'
    else:
        raise Exception('Wrong Evaluation Setup.')

    if evaluation_setup == 'e2e':
        # first generate belief state
        bs_tensor, bs_mask = data.pad_batch(bs_batch)
        if is_cuda:
            bs_tensor = bs_tensor.cuda(device)
            bs_mask = bs_mask.cuda(device)
        batch_bs_text = model.batch_generate(bs_tensor, bs_mask, generate_mode='bs', max_decode_len=max_response_len)
        # the belief state sequence could be long
        batch_bs_restore_text = []
        for idx in range(batch_size):
            if reform_bs_and_act:
                one_bs_text = batch_bs_text[idx]
                res_batch_parse_dict[idx]['bspn_gen_reform'] = one_bs_text
                one_bs_restore_text = restore_text(one_bs_text, mode='bs')
                res_batch_parse_dict[idx]['bspn_gen'] = one_bs_restore_text
                batch_bs_restore_text.append(one_bs_restore_text)
            else:
                one_bs_text = batch_bs_text[idx]
                res_batch_parse_dict[idx]['bspn_gen'] = one_bs_text
        if reform_bs_and_act:
            batch_bs_text = batch_bs_restore_text
        else:
            pass

        if input_contain_db:
            # we need to query the db base
            batch_db_input_id_list = []
            for idx in range(batch_size):
                one_queried_db_result = \
                data.reader.bspan_to_DBpointer(batch_bs_text[idx], res_batch_parse_dict[idx]['turn_domain'])
                one_db_text = '<sos_db> ' + one_queried_db_result + ' <eos_db>' 
                #print (db_text)
                #print (one_db_text)
                one_db_token_id_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_db_text))
                batch_db_input_id_list.append(one_db_token_id_input)
        else:
            batch_db_input_id_list = []
            for _ in range(batch_size):
                batch_db_input_id_list.append([])

        # then we generate the dialogue action
        da_batch_input_id_list = []
        for idx in range(batch_size):
            one_da_input_id_list = da_batch[idx] + batch_db_input_id_list[idx]
            da_batch_input_id_list.append(one_da_input_id_list)
        da_tensor, da_mask = data.pad_batch(da_batch_input_id_list)
        if is_cuda:
            da_tensor = da_tensor.cuda(device)
            da_mask = da_mask.cuda(device)
        batch_da_text = model.batch_generate(da_tensor, da_mask, generate_mode='da', max_decode_len=max_span_len)
        for idx in range(batch_size):
            if reform_bs_and_act:
                res_batch_parse_dict[idx]['aspn_gen_reform'] = batch_da_text[idx]
                res_batch_parse_dict[idx]['aspn_gen'] = restore_text(batch_da_text[idx], mode='da')
            else:
                res_batch_parse_dict[idx]['aspn_gen'] = batch_da_text[idx]            

        # finally, we generate the response
        nlg_batch_input_id_list = []
        for idx in range(batch_size):
            one_nlg_input_id_list = nlg_batch[idx] + batch_db_input_id_list[idx]
            nlg_batch_input_id_list.append(one_nlg_input_id_list)
        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch_input_id_list)
        if is_cuda:
            nlg_tensor = nlg_tensor.cuda(device)
            nlg_mask = nlg_mask.cuda(device)
        batch_nlg_text = model.batch_generate(nlg_tensor, nlg_mask, generate_mode='nlg', max_decode_len=max_response_len)
        for idx in range(batch_size):
            res_batch_parse_dict[idx]['resp_gen'] = batch_nlg_text[idx]

    elif evaluation_setup == 'policy': 
        # we need to generate the dialogue action and dialogue response
        # the da input already contains the ref db result
        da_tensor, da_mask = data.pad_batch(da_batch)
        if is_cuda:
            da_tensor = da_tensor.cuda(device)
            da_mask = da_mask.cuda(device)
        batch_da_text = model.batch_generate(da_tensor, da_mask, generate_mode='da', max_decode_len=max_span_len)
        for idx in range(batch_size):
            res_batch_parse_dict[idx]['aspn_gen'] = batch_da_text[idx]

        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch)
        if is_cuda:
            nlg_tensor = nlg_tensor.cuda(device)
            nlg_mask = nlg_mask.cuda(device)
        batch_nlg_text = model.batch_generate(nlg_tensor, nlg_mask, generate_mode='nlg', max_decode_len=max_response_len)
        for idx in range(batch_size):
            res_batch_parse_dict[idx]['resp_gen'] = batch_nlg_text[idx]

    elif evaluation_setup == 'oracle':
        # we only need to generate the response
        # nlg_batch already contains the ref db result
        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch)
        if is_cuda:
            nlg_tensor = nlg_tensor.cuda(device)
            nlg_mask = nlg_mask.cuda(device)
        batch_nlg_text = model.batch_generate(nlg_tensor, nlg_mask, generate_mode='nlg', max_decode_len=max_response_len)
        for idx in range(batch_size):
            res_batch_parse_dict[idx]['resp_gen'] = batch_nlg_text[idx]
    else:
        raise Exception('Wrong Evaluation Setup.')
    return res_batch_parse_dict
