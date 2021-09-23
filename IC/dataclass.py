import sys
import json
import torch
import random
import progressbar
import numpy as np
from torch.nn.utils import rnn

class Banking77:
    def __init__(self, tokenizer, train_path, test_path, datapoints_per_intent, format_mode):
        self.tokenizer = tokenizer
        print ('Tokenizer size is {}'.format(len(tokenizer)))

        if format_mode == 'bs':
            prefix_text = 'translate dialogue to belief state:'
            self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_b>'])[0]
            self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_b>'])[0]
        elif format_mode == 'da':
            prefix_text = 'translate dialogue to dialogue action:'
            self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]
            self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]
        elif format_mode == 'resp':
            prefix_text = 'translate dialogue to system response:'
            self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_r>'])[0]
            self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_r>'])[0]
        elif format_mode == 'ic':
            prefix_text = 'translate dialogue to user intent:'
            self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_d>'])[0]
            self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_d>'])[0]
        else:
            raise Exception('Wrong Format Mode!!!')
        self.prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prefix_text))

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_usr_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_u>'])[0]
        self.eos_usr_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_u>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
        self.datapoints_per_intent = datapoints_per_intent
        print ('Loading training data...')
        self.train_intent_uttr_dict = self.load_path(train_path)
        print ('Reforming training data with each intent has maximum {} datapoints'.format(self.datapoints_per_intent))
        self.train_data_id_list = self.reform_intent_uttr_dict(self.train_intent_uttr_dict, mode='train')
        print ('Training data size is {}'.format(len(self.train_data_id_list)))
        print ('Loading test data...')
        self.test_intent_uttr_dict = self.load_path(test_path)
        self.test_data_id_list = self.reform_intent_uttr_dict(self.test_intent_uttr_dict, mode='test')
        print ('Test data size is {}'.format(len(self.test_data_id_list)))
        self.train_num, self.test_num = len(self.train_data_id_list), len(self.test_data_id_list)

    def load_path(self, path):
        intent_utterance_dict = {}
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()[1:] # the first line is "text,category"
            for l in lines:
                item_list = l.strip('\n').split(',')
                try:
                    assert len(item_list) >= 2
                except:
                    continue
                text = ','.join(item_list[:-1]).strip()
                label = item_list[-1].strip()
                #text, label = item_list[0].strip(), item_list[1].strip()
                label = '[' + label.strip('[').strip(']') + ']'
                try:
                    intent_utterance_dict[label].append(text)
                except KeyError:
                    intent_utterance_dict[label] = [text]
        print ('Print file statistics...')
        for key in intent_utterance_dict:
            print ('The number of {} instances is {}'.format(key, len(intent_utterance_dict[key])))
        return intent_utterance_dict

    def tokenize_usr_text(self, text):
        token_id_list = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        token_id_list = [self.sos_context_token_id, self.sos_usr_token_id] + token_id_list + [self.eos_usr_token_id, self.eos_context_token_id]
        token_id_list = self.prefix_id + token_id_list
        return token_id_list

    def tokenize_label_text(self, text):
        token_id_list = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        token_id_list = [self.tgt_sos_token_id] + token_id_list + [self.tgt_eos_token_id]
        return token_id_list

    def reform_intent_uttr_dict(self, intent_uttr_dict, mode):
        data_id_list = []
        for intent in intent_uttr_dict:
            if mode == 'train':
                one_tmp_list = intent_uttr_dict[intent]
                import random
                random.shuffle(one_tmp_list)
                usr_text_list = one_tmp_list[:self.datapoints_per_intent] # randomly select a subset of training examples
                #usr_text_list = intent_uttr_dict[intent][:self.datapoints_per_intent]
            elif mode == 'test':
                usr_text_list = intent_uttr_dict[intent]
            else:
                raise Exception('Wrong dataset mode!!!')
            for text in usr_text_list:
                one_uttr_id_list = self.tokenize_usr_text(text.strip())
                one_label_id_list = self.tokenize_label_text(intent)
                data_id_list.append((one_uttr_id_list, one_label_id_list))
        return data_id_list

    def get_batches(self, batch_size, mode):
        #batch_size = self.cfg.batch_size
        batch_list = []
        if mode == 'train':
            random.shuffle(self.train_data_id_list)
            all_data_list = self.train_data_id_list
        elif mode == 'test':
            all_data_list = self.test_data_id_list
        else:
            raise Exception('Wrong Mode!!!')

        all_input_data_list, all_output_data_list = [], []
        for item in all_data_list:
            all_input_data_list.append(item[0])
            all_output_data_list.append(item[1])

        data_num = len(all_input_data_list)
        batch_num = int(data_num/batch_size) + 1

        for i in range(batch_num):
            start_idx, end_idx = i*batch_size, (i+1)*batch_size
            if start_idx > data_num - 1:
                break
            end_idx = min(end_idx, data_num - 1)
            one_input_batch_list, one_output_batch_list = [], []
            for idx in range(start_idx, end_idx):
                one_input_batch_list.append(all_input_data_list[idx])
                one_output_batch_list.append(all_output_data_list[idx])
            one_batch = [one_input_batch_list, one_output_batch_list]
            if len(one_batch[0]) == 0:
                pass
            else:
                batch_list.append(one_batch)
        print ('Number of {} batches is {}'.format(mode, len(batch_list)))
        return batch_list

    def build_iterator(self, batch_size, mode):
        batch_list = self.get_batches(batch_size, mode)
        for i, batch in enumerate(batch_list):
            yield batch

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list)
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels


