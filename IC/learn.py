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
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_prefix', type=str, help='the path where stores the data.')
    parser.add_argument('--datapoints_per_intent', type=int, help='number of training data for each intent (used for few-shot learning)')
    parser.add_argument('--format_mode', type=str, help='determines the input prefix. bs, da, or resp')
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')
    parser.add_argument('--pretrained_path', type=str, help='Pretrained checkpoint path.')
    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=64, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=1, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation step.")
    parser.add_argument("--save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--optimizer_name", type=str, default='adafactor', help="use which optimizer to train the model.")
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
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
 
    args = parse_config()
    device = torch.device('cuda')

    print ('Start loading data...')
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)

    from dataclass import Banking77
    train_path, test_path = args.data_prefix + '/train.csv', args.data_prefix + '/test.csv'
    data = Banking77(tokenizer, train_path, test_path, args.datapoints_per_intent, args.format_mode)
    print ('Data Loaded.')

    print ('Start loading model...')
    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(args.pretrained_path, tokenizer, args.format_mode, dropout=args.dropout)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded')

    # organize optimizer
    overall_batch_size = args.number_of_gpu * args.batch_size_per_gpu * args.gradient_accumulation_steps
    t_total = data.train_num * args.num_train_epochs // overall_batch_size
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule
        print ('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
            )
    elif args.optimizer_name == 'adam':
        print ('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        raise Exception('Wrong Optimizer Name!!!')

    optimizer.zero_grad()

    global_step = 0
    best_dev_acc = 0.
    for epoch in range(args.num_train_epochs):
        model.train()
        # --- training --- #
        print ('-----------------------------------------')
        print ('Start training at epoch %d' % epoch)
        train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train')
        train_batch_num_per_epoch = int(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))
        p = progressbar.ProgressBar(train_batch_num_per_epoch)
        p.start()
        p_train_idx = 0
        epoch_step, train_loss = 0, 0.
        for _, train_batch in enumerate(train_iterator):
            p.update(p_train_idx)
            p_train_idx += 1
            one_train_input_batch, one_train_output_batch = train_batch
            if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
            train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
            data.parse_batch_tensor(train_batch)
            if cuda_available:
                train_batch_src_tensor = train_batch_src_tensor.to(device)
                train_batch_src_mask = train_batch_src_mask.to(device)
                train_batch_input = train_batch_input.to(device)
                train_batch_labels = train_batch_labels.to(device)
            loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
            loss = loss.mean()
            #print (loss.size())
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_step += 1
            if (epoch_step+1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1) == train_batch_num_per_epoch:
                optimizer.step()
                if args.optimizer_name == 'adam':
                    scheduler.step() # only update learning rate for adam optimizer
                optimizer.zero_grad()
                global_step += 1
        p.finish()
        train_loss = train_loss / train_batch_num_per_epoch
        print ('At epoch {}, total update steps is {}, the total training loss is {}'.format(epoch, global_step, train_loss))
        print ('++++++++++++++++++++++++++++++++++++++++++')

        # evaluate the model after each training epoch
        print ('-----------------------------------------')
        print ('Start evaluation at global update step {}'.format(global_step))
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
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                model_save_path = args.save_path + '/epoch_{}_dev_acc_{}'.format(epoch, round(dev_acc, 4))
                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)

                if multi_gpu_training:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)
                print ('Model saved.')
                dev_pred_save_path = model_save_path + '/dev_predicted_result.txt'
                with open(dev_pred_save_path, 'w', encoding = 'utf8') as o:
                    for text in dev_pred_text_list:
                        o.writelines(text + '\n')
                dev_reference_save_path = model_save_path + '/dev_reference_result.txt'
                with open(dev_reference_save_path, 'w', encoding = 'utf8') as o:
                    for text in dev_reference_text_list:
                        o.writelines(text + '\n')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = args.save_path
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('epoch'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                max_save_num = 1
                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                        print (one_folder_name)
                        os.system('rm -r ' + one_folder_name)
                print ('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
            print ('current dev acc is {}, maximum dev acc is {}'.format(round(dev_acc,4), round(best_dev_acc,4)))
            global_step += 1


        model.train()
        print ('dev evaluation finished.')
        print ('Resume training....')
        print ('-----------------------------------------')