import os
import random
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
from eval import MultiWozEvaluator


def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    for name in file_names:
        if name.startswith('epoch'):
            print (name)
            return name

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')

    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")

    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')

    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    # training configuration
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--output_save_path", type=str, help="directory to save the model output.")
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
    from dataclass import MultiWozData
    from config import Config
    cfg = Config(args.data_path_prefix)
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer

    if args.pretrained_path != 'None':
        ckpt_name = get_checkpoint_name(args.pretrained_path)
        pretrained_path = args.pretrained_path + '/' + ckpt_name

    if args.pretrained_path != 'None':
        print ('Loading Pretrained Tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

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

    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', use_db_as_input=use_db_as_input, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=0.01)

    print ('Data loaded')
    evaluator = MultiWozEvaluator(data.reader, cfg)

    print ('Start loading model...')
    assert args.model_name.startswith('t5')
    from modelling.T5Model import T5Gen_Model
    if args.pretrained_path != 'None':
        model = T5Gen_Model(pretrained_path, data.tokenizer, data.special_token_list, dropout=0.0, 
            add_special_decoder_token=add_special_decoder_token, is_training=True)
    else:
        model = T5Gen_Model(args.model_name, data.tokenizer, data.special_token_list, dropout=0.0, 
            add_special_decoder_token=add_special_decoder_token, is_training=True)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    model.eval()
    print ('Model loaded')

    from e2e_inference_utlis import e2e_batch_generate
    with torch.no_grad():
        ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
        input_contain_db=use_db_as_input
        dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
            eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='test')
        dev_batch_num_per_epoch = len(dev_batch_list)
        p = progressbar.ProgressBar(dev_batch_num_per_epoch)
        print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
        p.start()
        all_dev_result = []
        for p_dev_idx in range(dev_batch_num_per_epoch):
            p.update(p_dev_idx)
            one_inference_batch = dev_batch_list[p_dev_idx]
            dev_batch_parse_dict = e2e_batch_generate(model, one_inference_batch, input_contain_db, data)
            for item in dev_batch_parse_dict:
                all_dev_result.append(item)
        p.finish()
        dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result)
        dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
        print ('The evaluation results are: Inform: {}, Success: {}, BLEU: {}, Combined Score: {}'.format(dev_match, 
            dev_success, dev_bleu, dev_score))
        one_dev_str = 'inference_result_e2e_evaluation_inform_{}_success_{}_bleu_{}_combine_score_{}'.format(round(dev_match, 2),
                round(dev_success,2), round(dev_bleu,2), round(dev_score,2))

        output_save_path = args.output_save_path + '/' + one_dev_str + '.json'
        import os
        if os.path.exists(args.output_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(args.output_save_path, exist_ok=True)

        import json
        with open(output_save_path, 'w') as outfile:
            json.dump(all_dev_result, outfile, indent=4)

