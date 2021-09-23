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
    parser.add_argument('--shuffle_mode', type=str, default='session_level', 
        help="session_level or turn_level, it controls how we shuffle the training data.")

    # pretraining datasets specification
    parser.add_argument('--use_nlu', type=str, default='True', help='whether using NLU data during pretraining.')
    parser.add_argument('--use_bs', type=str, default='True', help='whether using DST data during pretraining.')
    parser.add_argument('--use_da', type=str, default='True', help='whether using POL data during pretraining.')
    parser.add_argument('--use_nlg', type=str, default='True', help='whether using NLG data during pretraining.')
    parser.add_argument('--dataset_prefix_path', type=str, help='the path where all datasets are stored.')

    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')

    # training configuration
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--save_ckpt_name", type=str, help="the name under which to save the pre-trained model. small or base or large")
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
    preprocessed_tokenizer_path = args.dataset_prefix_path + r'/tokenizer_with_special_token/'
    tokenizer = T5Tokenizer.from_pretrained(preprocessed_tokenizer_path)

    from dataclass import TOD_PRETRAINING_CORPUS
    data = TOD_PRETRAINING_CORPUS(tokenizer, args.shuffle_mode, args.dataset_prefix_path, use_nlu=args.use_nlu, 
        use_bs=args.use_bs, use_da=args.use_da, use_nlg=args.use_nlg, max_tgt_len=128)
    print ('Data Loaded.')
    print ('Start loading model...')
    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(args.model_name, tokenizer, dropout=args.dropout)

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
    actual_total_steps = int((data.train_num * args.num_train_epochs) / overall_batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    print ('Use AdamW Optimizer for Training.')
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=actual_total_steps)
    optimizer.zero_grad()

    global_step = 0
    best_dev_loss = 1e10
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
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_step += 1
            if (epoch_step+1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1) == train_batch_num_per_epoch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step == actual_total_steps+1:
                    print ('Pretraining completed at steps {}'.format(global_step))
                    break

            if global_step > 0 and global_step % args.save_steps == 0:
                print ('-----------------------------------------')
                print ('Start evaluation at global update step {}'.format(global_step))
                model.eval()
                with torch.no_grad():
                    dev_batch_list = data.get_batches(args.number_of_gpu * args.batch_size_per_gpu, mode='dev')
                    dev_batch_num_per_epoch = len(dev_batch_list)
                    dev_p = progressbar.ProgressBar(dev_batch_num_per_epoch)
                    print ('Number of evaluation batches is {}'.format(dev_batch_num_per_epoch))
                    dev_p.start()
                    dev_loss = 0.
                    for p_dev_idx in range(dev_batch_num_per_epoch):
                        dev_p.update(p_dev_idx)
                        one_dev_batch = dev_batch_list[p_dev_idx]
                        if len(one_dev_batch[0]) == 0:
                            break
                        dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels = \
                        data.parse_batch_tensor(one_dev_batch)
                        if cuda_available:
                            dev_batch_src_tensor = dev_batch_src_tensor.to(device)
                            dev_batch_src_mask = dev_batch_src_mask.to(device)
                            dev_batch_input = dev_batch_input.to(device)
                            dev_batch_labels = dev_batch_labels.to(device)
                        one_dev_loss = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels)
                        one_dev_loss = one_dev_loss.mean()
                        dev_loss += one_dev_loss.item()
                    dev_loss /= dev_batch_num_per_epoch
                    print ('current dev loss is {}, minimum dev loss is {}'.format(round(dev_loss,2), round(best_dev_loss,2)))
                    if dev_loss < best_dev_loss:
                        # saving the model with the lowest validation perplexity
                        print ('Saving model...')
                        model_save_path = args.save_path + args.save_ckpt_name

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

                        # --------------------------------------------------------------------------------------------- #
                        # removing extra checkpoints...
                        # only save 1 checkpoints
                        import os
                        from operator import itemgetter
                        fileData = {}
                        test_output_dir = args.save_path
                        for fname in os.listdir(test_output_dir):
                            if fname.startswith(args.save_ckpt_name):
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
                    best_dev_loss = min(dev_loss, best_dev_loss)
                    global_step += 1
                model.train()
                print ('dev evaluation finished.')
                print ('Resume training....')
                print ('-----------------------------------------')
        p.finish()
        train_loss = train_loss / train_batch_num_per_epoch
        print ('At epoch {}, total update steps is {}, the total training loss is {}'.format(epoch, global_step, train_loss))
        print ('++++++++++++++++++++++++++++++++++++++++++')
