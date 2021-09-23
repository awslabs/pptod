import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
import os

class T5Gen_Model(nn.Module):
    def __init__(self, model_path, tokenizer, special_token_list, dropout, add_special_decoder_token, is_training):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.special_token_list = special_token_list
        self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
        self.sos_r_token_id, self.eos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>', 
            '<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>','<eos_r>'])
        if is_training:
            print ('Initializing Huggingface T5 model...')
            t5_config = T5Config.from_pretrained(model_path)
            t5_config.__dict__["dropout"] = dropout
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
        else:
            print ('Loading Model from pretrained ckpt...')
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.add_special_decoder_token = add_special_decoder_token

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input, labels=tgt_output)
        loss = outputs[0]#.mean()
        return loss

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

    def batch_generate(self, src_input, src_mask, generate_mode, max_decode_len):
        '''
            This function deals with batch generation. In order to fully take advantage of batch inference,
            in each batch, we only generate one type of output. e.g. Given a batch of dialogue history, we 
            generate the corresponding belief state/dialogue action/system response for the given batch. The 
            specific type of output is decided by the input argument "generate_mode"
        '''
        if self.add_special_decoder_token:
            if generate_mode == 'bs':
                start_token, end_token, start_token_id, end_token_id = '<sos_b>', '<eos_b>', self.sos_b_token_id, self.eos_b_token_id
            elif generate_mode == 'da':
                start_token, end_token, start_token_id, end_token_id = '<sos_a>', '<eos_a>', self.sos_a_token_id, self.eos_a_token_id
            elif generate_mode == 'nlg':
                start_token, end_token, start_token_id, end_token_id = '<sos_r>', '<eos_r>', self.sos_r_token_id, self.eos_r_token_id
            else:
                raise Exception('Wrong Generate Mode!!!')
        else:
            start_token, end_token = '<pad>', '</s>'
            start_token_id, end_token_id = \
            self.tokenizer.convert_tokens_to_ids([start_token])[0], self.tokenizer.convert_tokens_to_ids([end_token])[0]

        outputs = self.model.generate(input_ids = src_input, attention_mask = src_mask, decoder_start_token_id = start_token_id,
            pad_token_id = self.pad_token_id, eos_token_id = end_token_id, max_length = max_decode_len)

        res_text_list = []
        for predicted_ids in outputs:
            one_res_text = self.tokenized_decode(predicted_ids)
            #print (one_res_text)
            one_res_text = one_res_text.split(start_token)[-1].split(end_token)[0].strip()

            final_res_list = []
            for token in one_res_text.split():
                if token == '<_PAD_>':
                    continue
                else:
                    final_res_list.append(token)
            one_res_text = ' '.join(final_res_list).strip()
            
            res_text_list.append(one_res_text)
        return res_text_list

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)
        