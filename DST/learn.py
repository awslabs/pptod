import os
import sys
import json
import torch
import random
import argparse
import operator
import progressbar
import torch.nn as nn
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

def zip_result(prediction):
    result = {}
    for turn in prediction:
        dial_id = turn['dial_id']
        turn_idx = turn['turn_num']
        try:
            result[dial_id][turn_idx] = turn
        except KeyError:
            result[dial_id] = {}
            result[dial_id][turn_idx] = turn
    return result

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')

    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")

    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-base or t5-large or facebook/bart-base or facebook/bart-large')

    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')

    # training configuration
    parser.add_argument('--optimizer_name', default='adafactor', type=str, help='which optimizer to use during training, adam or adafactor')
    parser.add_argument('--specify_adafactor_lr', type=str, default='True', help='True or False, whether specify adafactor lr')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--epoch_num", default=60, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

def get_optimizers(model, args, train_num, optimizer_name, specify_adafactor_lr):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    overall_batch_size = args.number_of_gpu * args.batch_size_per_gpu * args.gradient_accumulation_steps
    num_training_steps = train_num * args.epoch_num // overall_batch_size
    print ('----------')
    if optimizer_name == 'adam':
        print ('Use Adam Optimizer for Training.')
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    elif optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule
        print ('Use Adafactor Optimizer for Training.')
        if specify_adafactor_lr:
            print ('Specific learning rate.')
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
            scheduler = None
        else:
            print ('Do not specific learning rate.')
            optimizer = Adafactor(optimizer_grouped_parameters, 
                scale_parameter=True, 
                relative_step=True, 
                warmup_init=True, 
                lr=None)
            scheduler = AdafactorSchedule(optimizer)
    else:
        raise Exception('Wrong Optimizer Name!!!')
    print ('----------')
    return optimizer, scheduler

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
    if args.pretrained_path != 'None':
        print ('Loading Pretrained Tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    if args.add_prefix == 'True':
        add_prefix = True
    elif args.add_prefix == 'False':
        add_prefix = False
    else:
        raise Exception('Wrong Prefix Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    if args.specify_adafactor_lr == 'True':
        specify_adafactor_lr = True
    elif args.specify_adafactor_lr == 'False':
        specify_adafactor_lr = False
    else:
        raise Exception('Wrong Specify LR Mode!!!')

    from dataclass import DSTMultiWozData
    data = DSTMultiWozData(args.model_name, tokenizer, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
                          data_mode='train', train_data_ratio=args.train_data_ratio)

    print ('Start loading model...')
    if args.model_name.startswith('facebook/bart'):
        # load bart model
        from modelling.BARTModel import BARTGen_Model
        if args.pretrained_path != 'None':
            model = BARTGen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=args.dropout, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)
        else:
            model = BARTGen_Model(args.model_name, data.tokenizer, data.special_token_list, dropout=args.dropout, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)
    elif args.model_name.startswith('t5'):
        from modelling.T5Model import T5Gen_Model
        if args.pretrained_path != 'None':
            model = T5Gen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=args.dropout, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)
        else:
            model = T5Gen_Model(args.model_name, data.tokenizer, data.special_token_list, dropout=args.dropout, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)
    else:
        raise Exception('Wrong Model Type!!!')

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded')

    optimizer, scheduler = get_optimizers(model, args, data.train_num, args.optimizer_name, specify_adafactor_lr)
    optimizer.zero_grad()

    min_dev_loss = 1e10
    max_dev_score, max_dev_str = 0., ''
    for epoch in range(args.epoch_num):
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
                if args.optimizer_name == 'adafactor' and not specify_adafactor_lr:
                    scheduler.step()
                elif args.optimizer_name == 'adam':
                    scheduler.step() # only update learning rate when using adam
                else:
                    pass
                optimizer.zero_grad()
        p.finish()
        train_loss = train_loss / train_batch_num_per_epoch
        print ('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
        print ('++++++++++++++++++++++++++++++++++++++++++')
        # **********************************************************************
        # --- evaluation --- #

        if args.train_data_ratio <= 0.1:
            if epoch < 5: # first train 10 epoches
                continue
        elif args.train_data_ratio == 0.2:
            if epoch < 3: # first train 5 epoches
                continue
        else:
            pass

        from inference_utlis import batch_generate
        print ('Start evaluation at epoch %d' % epoch)
        model.eval()
        with torch.no_grad():
            dev_batch_list = \
            data.build_all_evaluation_batch_list(eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='dev')
            dev_batch_num_per_epoch = len(dev_batch_list)
            p = progressbar.ProgressBar(dev_batch_num_per_epoch)
            print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
            p.start()
            all_dev_result = []
            for p_dev_idx in range(dev_batch_num_per_epoch):
                p.update(p_dev_idx)
                one_inference_batch = dev_batch_list[p_dev_idx]
                dev_batch_parse_dict = batch_generate(model, one_inference_batch, data)
                for item in dev_batch_parse_dict:
                    all_dev_result.append(item)
            p.finish()

            from compute_joint_acc import compute_jacc
            all_dev_result = zip_result(all_dev_result)
            dev_score = compute_jacc(data=all_dev_result) * 100
            one_dev_str = 'dev_joint_accuracy_{}'.format(round(dev_score,2))
            if dev_score > max_dev_score:
                max_dev_str = one_dev_str
                max_dev_score = dev_score
                print ('Saving Model...')
                model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str

                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)

                if cuda_available and torch.cuda.device_count() > 1:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)

                import json
                pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                with open(pkl_save_path, 'w') as outfile:
                    json.dump(all_dev_result, outfile, indent=4)

                #import pickle
                #pickle.dump(all_dev_result, open(pkl_save_path, "wb"))

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                # only save 2 checkpoints
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = args.ckpt_save_path
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
            print ('Currnt joint accuracy is {}, best joint accuracy is {}'.format(round(dev_score, 2), round(max_dev_score, 2)))

            print ('Current Result: ' + one_dev_str)
            print ('Best Result: ' + max_dev_str)

            print ('dev evaluation finished.')
        print ('-----------------------------------------')
        model.train()

