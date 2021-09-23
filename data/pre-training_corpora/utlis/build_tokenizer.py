if __name__ == '__main__':
    import os
    save_path = r'../tokenized_pretraining_corpora/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)
    
    sos_eos_tokens = ['<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<go_d>','<eos_d>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', 
                '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>']
    # initialize tokenizer
    print ('Saving Tokenizer...')
    from transformers import T5Tokenizer
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(sos_eos_tokens)
    tokenizer_save_path = save_path + r'/tokenizer_with_special_token'
    tokenizer.save_pretrained(tokenizer_save_path)
    print ('Tokenizer Saved.')
