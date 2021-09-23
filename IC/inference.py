import os
import sys
import time
import json
import torch
import random
import argparse
import operator
import progressbar
import numpy as np
import torch.nn as nn
from operator import itemgetter
import torch.nn.functional as F

def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    for name in file_names:
        if name.startswith('epoch'):
            print (name)
            return name

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_prefix', type=str, help='the path where stores the data.')
    parser.add_argument('--format_mode', type=str, help='bs or ic')
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')
    parser.add_argument('--pretrained_path', type=str, help='pre-trained checkpoint path.')
    # training configuration
    parser.add_argument("--batch_size_per_gpu", type=int, default=64, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=1, help="Number of available GPUs.")  
    parser.add_argument("--save_path", type=str, help='Path to store the predicted result.')
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU.')
    else:
        pass
 
    args = parse_config()
    device = torch.device('cuda')

    print ('Start loading data...')
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    ckpt_name = get_checkpoint_name(args.pretrained_path)
    pretrained_path = args.pretrained_path + '/' + ckpt_name
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path)

    from dataclass import Banking77
    train_path, test_path = args.data_prefix + '/train.csv', args.data_prefix + '/test.csv'
    data = Banking77(tokenizer, train_path, test_path, datapoints_per_intent=10, format_mode=args.format_mode)
    print ('Data Loaded.')

    print ('Start loading model...')
    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(pretrained_path, tokenizer, format_mode=args.format_mode, dropout=0.0)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded')

    print ('-----------------------------------------')
    print ('Start evaluation...')
    model.eval()
    with torch.no_grad():
        dev_batch_list = data.get_batches(args.number_of_gpu * args.batch_size_per_gpu, mode='test')
        dev_batch_num_per_epoch = len(dev_batch_list)
        dev_p = progressbar.ProgressBar(dev_batch_num_per_epoch)
        print ('Number of evaluation batches is {}'.format(dev_batch_num_per_epoch))
        dev_p.start()
        dev_pred_text_list, dev_reference_text_list = [], []
        for p_dev_idx in range(dev_batch_num_per_epoch):
            dev_p.update(p_dev_idx)
            one_dev_batch = dev_batch_list[p_dev_idx]
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels = data.parse_batch_tensor(one_dev_batch)
            if cuda_available:
                dev_batch_src_tensor = dev_batch_src_tensor.to(device)
                dev_batch_src_mask = dev_batch_src_mask.to(device)
                dev_batch_input = dev_batch_input.to(device)
                dev_batch_labels = dev_batch_labels.to(device)
            if multi_gpu_training:
                one_dev_prediction_text_list = model.module.batch_prediction(dev_batch_src_tensor, dev_batch_src_mask)
            else:
                one_dev_prediction_text_list = model.batch_prediction(dev_batch_src_tensor, dev_batch_src_mask)
            dev_pred_text_list += one_dev_prediction_text_list
            if multi_gpu_training:
                dev_reference_text_list += model.module.parse_batch_text(dev_batch_input)
            else:
                dev_reference_text_list += model.parse_batch_text(dev_batch_input)
        dev_p.finish()
        assert len(dev_pred_text_list) == len(dev_reference_text_list)
        dev_same_num = 0
        for eva_idx in range(len(dev_pred_text_list)):
            if dev_pred_text_list[eva_idx].strip() == dev_reference_text_list[eva_idx].strip():
                dev_same_num += 1
        dev_acc = 100 * (dev_same_num / len(dev_pred_text_list))
        print ('Inference accuracy is {}'.format(round(dev_acc,4)))

        import os
        save_path = args.save_path
        if os.path.exists(save_path):
            pass
        else: # recursively construct directory
            os.makedirs(save_path, exist_ok=True)

        dev_pred_save_path = save_path + '/predicted_labels.txt'
        with open(dev_pred_save_path, 'w', encoding = 'utf8') as o:
            for text in dev_pred_text_list:
                o.writelines(text + '\n')
        dev_reference_save_path = save_path + '/reference_labels.txt'
        with open(dev_reference_save_path, 'w', encoding = 'utf8') as o:
            for text in dev_reference_text_list:
                o.writelines(text + '\n')

        print ('Evaluation finished!')
        print ('-----------------------------------------')