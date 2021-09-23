import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import ontology
import random
from torch.nn.utils import rnn

all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class DSTMultiWozData:
    def __init__(self, model_name, tokenizer, data_path_prefix, shuffle_mode='shuffle_session_level', 
        data_mode='train', add_prefix=True, add_special_decoder_token=True, train_data_ratio=1.0):
        '''
            model_name: t5-small or t5-base or t5-large

            data_path_prefix: where the data stores

            shuffle_mode: turn level shuffle or session level shuffle

            add_prefix: whether adding task-specifc prompt to drive the generation

            add_special_decoder_token: whether add special decoder token for generation of each subtasks
                    <sos_b>, <eos_b> for belief state tracking
                    <sos_a>, <eos_a> for dialogue action prediction
                    <sos_r>, <eos_r> for response generation
        '''
        self.add_prefix = add_prefix
        assert self.add_prefix in [True, False]
        self.add_special_decoder_token = add_special_decoder_token
        assert self.add_special_decoder_token in [True, False]

        self.tokenizer = tokenizer
        print ('Original Tokenizer Size is %d' % len(self.tokenizer))
        self.special_token_list = self.add_sepcial_tokens()
        print ('Tokenizer Size after extension is %d' % len(self.tokenizer))
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]

        # initialize bos_token_id, eos_token_id
        self.model_name = model_name
        assert self.model_name.startswith('t5')
        from transformers import T5Config
        t5config = T5Config.from_pretrained(model_name)
        self.bos_token_id = t5config.decoder_start_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.bos_token = self.tokenizer.convert_ids_to_tokens([self.bos_token_id])[0]
        self.eos_token = self.tokenizer.convert_ids_to_tokens([self.eos_token_id])[0]
        print ('bos token is {}, eos token is {}'.format(self.bos_token, self.eos_token))

        self.all_sos_token_id_list = []
        for token in all_sos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            self.all_sos_token_id_list.append(one_id)
            print (self.tokenizer.convert_ids_to_tokens([one_id]))
        print (len(self.all_sos_token_id_list))
        self.all_eos_token_id_list = []
        for token in all_eos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            self.all_eos_token_id_list.append(one_id)
            print (self.tokenizer.convert_ids_to_tokens([one_id]))
        print (len(self.all_eos_token_id_list))

        if self.add_prefix:
            bs_prefix_text = 'translate dialogue to belief state:'
            self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
        else:
            self.bs_prefix_id = []

        import json
        if data_mode == 'train':
            train_json_path = data_path_prefix + '/multiwoz-fine-processed-train.json'
            with open(train_json_path) as f:
                train_raw_data = json.load(f)

            self.train_data_ratio = train_data_ratio
            assert self.train_data_ratio > 0
            # few-shot learning
            if self.train_data_ratio < 1.0:
                print ('Few-shot training setup.')
                few_shot_num = int(len(train_raw_data) * self.train_data_ratio) + 1
                random.shuffle(train_raw_data)
                # randomly select a subset of training data
                train_raw_data = train_raw_data[:few_shot_num]
                print ('Number of training sessions is {}'.format(few_shot_num))

            print ('Tokenizing raw train data...')
            train_data_id_list = self.tokenize_raw_data(train_raw_data)
            self.train_data_list = self.flatten_data(train_data_id_list)
            # record training data
            self.train_id2session_dict = {}
            self.train_dial_id_list = []
            for item in self.train_data_list:
                one_item_id = item['dial_id']
                try:
                    self.train_id2session_dict[one_item_id].append(item)
                except KeyError:
                    self.train_dial_id_list.append(one_item_id)
                    self.train_id2session_dict[one_item_id] = [item]
            assert len(self.train_dial_id_list) == len(self.train_id2session_dict)
            self.train_num = len(self.train_data_list) 
        elif data_mode == 'test':
            train_raw_data = []
        else:
            raise Exception('Wrong Data Mode!!!')

        dev_json_path = data_path_prefix + '/multiwoz-fine-processed-dev.json'
        with open(dev_json_path) as f:
            dev_raw_data = json.load(f)
        print ('Tokenizing raw dev data...')
        dev_data_id_list = self.tokenize_raw_data(dev_raw_data)
        self.dev_data_list = self.flatten_data(dev_data_id_list)

        test_json_path = data_path_prefix + '/multiwoz-fine-processed-test.json'
        with open(test_json_path) as f:
            test_raw_data = json.load(f)
        print ('Tokenizing raw test data...')
        test_data_id_list = self.tokenize_raw_data(test_raw_data)
        self.test_data_list = self.flatten_data(test_data_id_list)

        print ('The size of raw train, dev and test sets are %d, %d and %d' % \
            (len(train_raw_data), len(dev_raw_data), len(test_raw_data)))

        self.dev_num, self.test_num = len(self.dev_data_list), len(self.test_data_list)
        if data_mode == 'train':
            print ('train turn number is %d, dev turn number is %d, test turn number is %d' % \
                (len(self.train_data_list), len(self.dev_data_list), len(self.test_data_list)))
            self.shuffle_mode = shuffle_mode
            self.ordering_train_data()
        else:
            pass

    def ordering_train_data(self):
        if self.shuffle_mode == 'shuffle_turn_level':
            random.shuffle(self.train_data_list)
        elif self.shuffle_mode == 'shuffle_session_level':
            train_data_list = []
            random.shuffle(self.train_dial_id_list)
            for dial_id in self.train_dial_id_list:
                one_session_list = self.train_id2session_dict[dial_id]
                for one_turn in one_session_list:
                    train_data_list.append(one_turn)
            assert len(train_data_list) == len(self.train_data_list)
            self.train_data_list = train_data_list
        elif self.shuffle_mode == 'unshuffle':
            pass
        else:
            raise Exception('Wrong Train Ordering Mode!!!')

    def replace_sos_eos_token_id(self, token_id_list):
        if self.add_special_decoder_token: # if adding special decoder tokens, then no replacement
            sos_token_id_list = []
            eos_token_id_list = []
        else:
            sos_token_id_list = self.all_sos_token_id_list
            eos_token_id_list = self.all_eos_token_id_list

        res_token_id_list = []
        for one_id in token_id_list:
            if one_id in sos_token_id_list:
                res_token_id_list.append(self.bos_token_id)
            elif one_id in eos_token_id_list:
                res_token_id_list.append(self.eos_token_id)
            else:
                res_token_id_list.append(one_id)
        return res_token_id_list

    def tokenize_raw_data(self, raw_data_list):
        data_num = len(raw_data_list)
        p = progressbar.ProgressBar(data_num)
        p.start()
        all_session_list = []
        for idx in range(data_num):
            p.update(idx)
            one_sess_list = []
            for turn in raw_data_list[idx]:
                one_turn_dict = {}
                for key in turn:
                    if key in ['dial_id', 'pointer', 'turn_domain', 'turn_num', 'aspn', 'dspn', 'aspn_reform', 'db']:
                        one_turn_dict[key] = turn[key]
                    else:
                        # only tokenize ["user", "usdx", "resp", "bspn", "bsdx", "bspn_reform", "bsdx_reform"]
                        value_text = turn[key]
                        value_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_text))
                        value_id = self.replace_sos_eos_token_id(value_id)
                        one_turn_dict[key] = value_id
                one_sess_list.append(one_turn_dict)
            all_session_list.append(one_sess_list)
        p.finish()
        assert len(all_session_list) == len(raw_data_list)
        return all_session_list

    def shuffle_train_data(self):
        random.shuffle(self.train_data_list)

    def tokenized_decode(self, token_id_list):
        pred_tokens = self.tokenizer.convert_ids_to_tokens(token_id_list)
        res_text = ''
        curr_list = []
        for token in pred_tokens:
            if token in self.special_token_list + ['<s>', '</s>', '<pad>']:
                if len(curr_list) == 0:
                    res_text += ' ' + token + ' '
                else:
                    curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
                    res_text = res_text + ' ' + curr_res + ' ' + token + ' '
                    curr_list = []
            else:
                curr_list.append(token)
        if len(curr_list) > 0:
            curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
            res_text = res_text + ' ' + curr_res + ' '
        res_text_list = res_text.strip().split()
        res_text = ' '.join(res_text_list).strip()
        return res_text

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        special_tokens = ontology.sos_eos_tokens
        print (special_tokens)
        #print (special_tokens)
        self.tokenizer.add_tokens(special_tokens)
        return special_tokens

    def flatten_data(self, data):
        '''
            transform session data input turn data
            each item in session has length of (number of turns)
            each turn has the following keys: 
                dial_id: data id
                user: user input at this turn; 
                      e.g. '<sos_u> i am looking for an expensive restaurant in the centre . thank you . <eos_u>'
                usdx: delexicalized user input at this turn; 
                      e.g. '<sos_u> i am looking for an expensive restaurant in the centre . thank you . <eos_u>'
                resp: delexicialized system response; 
                      e.g. '<sos_r> there are several restaurant -s in the price range what type of food would you like to eat ? <eos_r>'
                bspn: belief state span;
                      e.g. '<sos_b> [restaurant] pricerange expensive area centre <eos_b>'
                bsdx: delexicialized belief state span;
                      e.g. '<sos_b> [restaurant] pricerange area <eos_b>'
                aspn: action span;
                      e.g. '<sos_a> [restaurant] [request] food <eos_a>'
                dspn: domain span;
                      e.g. '<sos_d> [restaurant] <eos_d>'
                pointer: e.g. [0, 0, 0, 1, 0, 0]
                turn_domain: e.g. ['[restaurant]']
                turn_num: the turn number in current session
                db: database result e.g. '<sos_db> [db_3] <eos_db>'
                bspn_reform: reformed belief state;
                      e.g. '<sos_b> [restaurant] pricerange = expensive , area = centre <eos_b>'
                bsdx_reform: reformed delexicialized belief state;
                      e.g. '<sos_b> [restaurant] pricerange , area <eos_b>'
                aspn_reform: reformed dialogue action;
                      e.g. '<sos_a> [restaurant] [request] food <eos_a>'
        '''
        data_list = []
        for session in data:
            one_dial_id = session[0]['dial_id']
            turn_num = len(session)
            previous_context = [] # previous context contains all previous user input and system response
            for turn_id in range(turn_num):
                curr_turn = session[turn_id]
                assert curr_turn['turn_num'] == turn_id # the turns should be arranged in order
                curr_user_input = curr_turn['user']
                curr_sys_resp = curr_turn['resp']
                curr_bspn = curr_turn['bspn']

                # construct belief state data
                bs_input = previous_context + curr_user_input
                bs_input = self.bs_prefix_id + [self.sos_context_token_id] + bs_input[-900:] + [self.eos_context_token_id]
                bs_output = curr_bspn

                data_list.append({'dial_id': one_dial_id,
                    'turn_num': turn_id,
                    'prev_context':previous_context,
                    'user': curr_turn['user'],
                    'usdx': curr_turn['usdx'],
                    'resp': curr_sys_resp,
                    'bspn': curr_turn['bspn'],
                    'bspn_reform': curr_turn['bspn_reform'],
                    'bsdx': curr_turn['bsdx'],
                    'bsdx_reform': curr_turn['bsdx_reform'],
                    'bs_input': bs_input,
                    'bs_output': bs_output
                    })
                # update context for next turn
                previous_context = previous_context + curr_user_input + curr_sys_resp
        return data_list

    def get_batches(self, batch_size, mode):
        batch_list = []
        if mode == 'train':
            data_num = self.train_num
            all_data_list = self.train_data_list
            self.ordering_train_data()
        elif mode == 'dev':
            data_num = self.dev_num
            all_data_list = self.dev_data_list
        elif mode == 'test':
            data_num = self.test_num
            all_data_list = self.test_data_list
        else:
            raise Exception('Wrong Mode!!!')

        all_input_data_list, all_output_data_list = [], []
        for item in all_data_list:
            one_input_data_list = []
            for key in ['bs_input']:
                one_input_data_list.append(item[key])
            all_input_data_list.extend(one_input_data_list)

            one_output_data_list = []
            for key in ['bs_output']:
                one_output_data_list.append(item[key])
            all_output_data_list.extend(one_output_data_list)

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
            batch_list.append(one_batch)
        out_str = 'Overall Number of datapoints is ' + str(data_num) + \
        ' Number of ' + mode + ' batches is ' + str(len(batch_list))
        print (out_str)
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
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list) # padded target sequence
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels

    def remove_sos_eos_token(self, text):
        token_list = text.split()
        res_list = []
        for token in token_list:
            if token == '<_PAD_>' or token.startswith('<eos_') or token.startswith('<sos_') or token in [self.bos_token, self.eos_token]:
                continue
            else:
                res_list.append(token)
        return ' '.join(res_list).strip()

    def parse_id_to_text(self, id_list):
        res_text = self.tokenized_decode(id_list)
        res_text = self.remove_sos_eos_token(res_text)
        return res_text

    def parse_one_eva_instance(self, one_instance):
        '''
            example data instance:
                {'dial_id': 'sng0547',
                 'turn_num': 0,
                 'user': 'i am looking for a high end indian restaurant, are there any in town?',
                 'bspn_gen': '[restaurant] food indian pricerange expensive',
                 'bsdx': '[restaurant] food pricerange',
                 'resp_gen': 'there are [value_choice] . what area of town would you like?',
                 'resp': 'there are [value_choice] [value_price] [value_food] restaurant -s in cambridge. is there an area of town that you prefer?',
                 'bspn': '[restaurant] food indian pricerange expensive',
                 'pointer': 'restaurant: >3; '}
            input_contain_db: whether input contain db result
            ref_db: if input contain db, whether using the reference db result
        '''
        res_dict = {}
        res_dict['dial_id'] = one_instance['dial_id']
        res_dict['turn_num'] = one_instance['turn_num']
        res_dict['user'] = self.parse_id_to_text(one_instance['user'])
        res_dict['bspn'] = self.parse_id_to_text(one_instance['bspn'])
        res_dict['bsdx'] = self.parse_id_to_text(one_instance['bsdx'])
        res_dict['bspn_reform'] = self.parse_id_to_text(one_instance['bspn_reform'])
        res_dict['bsdx_reform'] = self.parse_id_to_text(one_instance['bsdx_reform'])
        previous_context = one_instance['prev_context']
        curr_user_input = one_instance['user']

        # belief state setup
        res_dict['bspn_gen'] = ''
        bs_input_id_list = previous_context + curr_user_input
        bs_input_id_list = self.bs_prefix_id + [self.sos_context_token_id] + bs_input_id_list[-900:] + [self.eos_context_token_id]
        return bs_input_id_list, res_dict

    def build_batch_list(self, all_data_list, batch_size):
        data_num = len(all_data_list)
        batch_num = int(data_num/batch_size) + 1
        batch_list = []
        for i in range(batch_num):
            start_idx, end_idx = i*batch_size, (i+1)*batch_size
            if start_idx > data_num - 1:
                break
            end_idx = min(end_idx, data_num - 1)
            one_batch_list = []
            for idx in range(start_idx, end_idx):
                one_batch_list.append(all_data_list[idx])
            if len(one_batch_list) == 0: 
                pass
            else:
                batch_list.append(one_batch_list)
        return batch_list

    def build_all_evaluation_batch_list(self, eva_batch_size, eva_mode):
        '''
            ref_bs: whether using reference belief state to perform generation
                    if with reference belief state, then we also use reference db result
                    else generating belief state to query the db
            ref_act: whether using reference dialogue action to perform generation
                    if true: it always means that we also use reference belief state
                    if false: we can either use generated belief state and queried db result or
                              use reference belief state and reference db result
            eva_mode: 'dev' or 'test'; perform evaluation either on dev set or test set
            eva_batch_size: size of each evaluated batch
        '''
        if eva_mode == 'dev':
            data_list = self.dev_data_list
            eva_num = self.dev_num
        elif eva_mode == 'test':
            data_list = self.test_data_list
            eva_num = self.test_num
        else:
            raise Exception('Wrong Evaluation Mode!!!')

        all_bs_input_id_list, all_parse_dict_list = [], []
        for item in data_list:
            one_bs_input_id_list, one_parse_dict = self.parse_one_eva_instance(item)
            all_bs_input_id_list.append(one_bs_input_id_list)
            all_parse_dict_list.append(one_parse_dict)
        assert len(all_bs_input_id_list) == len(all_parse_dict_list)
        bs_batch_list = self.build_batch_list(all_bs_input_id_list, eva_batch_size)
        parse_dict_batch_list = self.build_batch_list(all_parse_dict_list, eva_batch_size)

        batch_num = len(bs_batch_list)
        final_batch_list = []
        for idx in range(batch_num):
            one_final_batch = [bs_batch_list[idx], parse_dict_batch_list[idx]]
            if len(bs_batch_list[idx]) == 0: 
                continue
            else:
                final_batch_list.append(one_final_batch)
        return final_batch_list
