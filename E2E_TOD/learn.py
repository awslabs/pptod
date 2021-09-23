import os
import sys
import json
import random
import torch
from torch import nn
import torch.nn.functional as F
import operator
from operator import itemgetter
import progressbar
import argparse
from eval import MultiWozEvaluator

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')

    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")

    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')

    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small, t5-base or t5-large')

    parser.add_argument('--pretrained_path', type=str, help='the path that stores pretrained checkpoint.')

    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--epoch_num", default=60, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    return parser.parse_args()


def get_optimizers(model, args):
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
    from transformers.optimization import Adafactor
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

    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    print ('Loading Pretrained Tokenizer...')
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    print ('Start loading data...')
    from dataclass import MultiWozData
    from config import Config
    cfg = Config(args.data_path_prefix)

    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', use_db_as_input=use_db_as_input, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio)
    print ('Data loaded')
    evaluator = MultiWozEvaluator(data.reader, cfg)

    print ('Start loading model...')
    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=args.dropout, 
        add_special_decoder_token=add_special_decoder_token, is_training=True)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded')

    optimizer, _ = get_optimizers(model, args)
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
                optimizer.zero_grad()
        p.finish()
        train_loss = train_loss / train_batch_num_per_epoch
        print ('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
        print ('++++++++++++++++++++++++++++++++++++++++++')
        # **********************************************************************
        # for few-shot learning, we let the model to first train for several epochs
        if args.train_data_ratio <= 0.1:
            if args.pretrained_path == 'None':
                if epoch < 10: # first train 10 epoches
                    continue
            else:
                if epoch < 3: # first train 10 epoches
                    continue
        elif args.train_data_ratio == 0.2:
            if epoch < 3: # first train 5 epoches
                continue
        else:
            pass
        # **********************************************************************
        # --- evaluation --- #
        from inference_utlis import batch_generate
        print ('Start evaluation at epoch %d' % epoch)
        model.eval()
        with torch.no_grad():
            ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
            input_contain_db=use_db_as_input
            dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
                eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='dev')
            dev_batch_num_per_epoch = len(dev_batch_list)
            p = progressbar.ProgressBar(dev_batch_num_per_epoch)
            print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
            p.start()
            all_dev_result = []
            for p_dev_idx in range(dev_batch_num_per_epoch):
                p.update(p_dev_idx)
                one_inference_batch = dev_batch_list[p_dev_idx]
                dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db, 
                    input_contain_db, data)
                for item in dev_batch_parse_dict:
                    all_dev_result.append(item)
            p.finish()
            dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result)

            dev_score = 0.5 * (dev_success + dev_match) + dev_bleu

            print ('Inform: %2.2f  Success: %2.2f  BLEU: %2.2f    Score: %.2f' % (dev_match, dev_success, dev_bleu, dev_score))
            one_dev_str = 'dev_e2e_evaluation_inform_{}_success_{}_bleu_{}_combine_score_{}'.format(round(dev_match, 2),
                round(dev_success,2), round(dev_bleu,2), round(dev_score,2))
            if dev_score > max_dev_score:
                max_dev_str = one_dev_str
                max_dev_score = dev_score
                print ('Saving Model...')
                model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str
                #model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'

                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)

                if cuda_available and torch.cuda.device_count() > 1:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)

                pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                with open(pkl_save_path, 'w') as outfile:
                    json.dump(all_dev_result, outfile, indent=4)
                print ('Validation result saved.')
                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                # only save 1 checkpoints
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

            print ('Current Result: ' + one_dev_str)
            print ('Best Result: ' + max_dev_str)
            print ('dev evaluation finished.')
        print ('-----------------------------------------')
        model.train()

