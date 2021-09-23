def parse_one_instance(text, tokenizer):
    item_list = text.strip('\n').split('\t')
    assert len(item_list) == 2
    text, label = item_list
    text = sos_u_token + ' ' + text + ' ' + eos_u_token
    label = sos_d_token + ' ' + label + ' ' + eos_d_token
    text_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    label_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label))
    return text, text_id, label, label_id

def process_file(in_f, tokenizer, out_f):
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
    res_list = []
    for l in lines:
        one_text, one_text_id, one_label, one_label_id = parse_one_instance(l, tokenizer)
        one_dict = {"user":one_text,
                   "user_id_list":one_text_id,
                   "intent":one_label,
                   "intent_id_list":one_label_id}
        res_list.append(one_dict)
        
    import json
    with open(out_f, 'w') as outfile:
        json.dump(res_list, outfile, indent=4)

if __name__ == '__main__':
    sos_u_token, eos_u_token = '<sos_u>', '<eos_u>'
    sos_d_token, eos_d_token = '<sos_d>', '<eos_d>'


    save_path = r'../tokenized_pretraining_corpora/'
    tokenizer_path = save_path + r'/tokenizer_with_special_token'
    print ('Loading tokenizer...')
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print ('Tokenizer loaded.')

    print ('Processing Intent Classification Datasets...')
    in_f = r'../separate_datasets/Intent_Classification/train_intent_classification_data.txt'
    out_f = save_path + r'/train_intent_classification.json'
    process_file(in_f, tokenizer, out_f)

    in_f = r'../separate_datasets/Intent_Classification/test_intent_classification_data.txt'
    out_f = save_path + r'/test_intent_classification.json'
    process_file(in_f, tokenizer, out_f)
    print ('Processing Intent Classification Datasets Finished!')

