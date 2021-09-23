import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
from dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2, paser_bs
import argparse

def compute_jacc(data,default_cleaning_flag=True,type2_cleaning_flag=False):
    num_turns = 0
    joint_acc = 0
    error = {}
    clean_tokens = ['<|endoftext|>', ]
    for file_name in data:
        for turn_id, turn_data in data[file_name].items():
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = paser_bs(turn_target)
            turn_pred = paser_bs(turn_pred)
            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)

            # MultiWOZ default cleaning
            if default_cleaning_flag:
                turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

            join_flag = False
            if set(turn_target) == set(turn_pred):
                joint_acc += 1
                join_flag = True
            
            elif type2_cleaning_flag: # check for possible Type 2 noisy annotations
                flag = True
                for bs in turn_target:
                    if bs not in turn_pred:
                        flag = False
                        break
                if flag:
                    for bs in turn_pred:
                        if bs not in turn_target:
                            flag = False
                            break

                if flag: # model prediction might be correct if found in Type 2 list of noisy annotations
                    dial_name = dial.split('.')[0]
                    if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]: # ignore these turns
                        pass
                    else:
                        joint_acc += 1
                        join_flag = True
            if not join_flag:
                if file_name not in error:
                    error[file_name] = {}
                turn_data['gtbs'] = turn_target
                turn_data['predbs'] = turn_pred
                error[file_name][turn_id] = turn_data
                
            num_turns += 1

    joint_acc /= num_turns
    
    print('joint accuracy: {}'.format(joint_acc))
    return joint_acc
    