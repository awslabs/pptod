import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
import os

class T5Gen_Model(nn.Module):
    def __init__(self, model_path, tokenizer, dropout):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
        self.sos_r_token_id, self.eos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>', 
            '<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>','<eos_r>'])

        print ('Initializing Huggingface T5 model...')
        t5_config = T5Config.from_pretrained(model_path)
        t5_config.__dict__["dropout"] = dropout
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input, labels=tgt_output)
        loss = outputs[0]#.mean()
        return loss

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)
        