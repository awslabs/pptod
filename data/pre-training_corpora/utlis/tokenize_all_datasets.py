import json
def tokenize_text(tokenizer, text, mode):
    if mode == 'bs':
        text = '<sos_b> ' + text + ' <eos_b>'
    elif mode == 'da':
        text = '<sos_a> ' + text + ' <eos_a>'
    elif mode == 'nlg':
        text = '<sos_r> ' + text + ' <eos_r>'
    elif mode == 'usr':
        text = '<sos_u> ' + text + ' <eos_u>'
    else:
        raise Exception('Wrong Mode!!!')
    text = ' '.join(text.split())
    text_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    return text, text_id_list

def process_one_dict(tokenizer, in_dict):
    res_dict = in_dict.copy()
    res_dict["dialogue_session"] = []
    for item in in_dict["dialogue_session"]:
        one_res_dict = {}
        for key in item:
            if key in ["turn_num", "turn_domain"]:
                one_res_dict[key] = item[key]
                continue
            elif key == "user":
                mode = 'usr'
            elif key == "resp":
                mode = 'nlg'
            elif key in ["bspn", "bsdx"]:
                mode = 'bs'
            elif key in ["aspn"]:
                mode = 'da'
            else: 
                raise Exception('Wrong Key!!!')
            id_key = key + '_id_list'
            text, text_id_list = tokenize_text(tokenizer, item[key], mode)
            one_res_dict[key] = text
            one_res_dict[id_key] = text_id_list
        res_dict["dialogue_session"].append(one_res_dict)
    return res_dict

import progressbar
import os
def process_file(path_prefix, file_name, tokenizer, output_path_prefix):
    print ('Start processing {}'.format(file_name))
    in_f = path_prefix + file_name
    with open(in_f) as f:
        data = json.load(f) 
    data_num = len(data)
    p = progressbar.ProgressBar(data_num)
    p.start()
    res_list = []
    for idx in range(data_num):
        p.update(idx)
        one_res_dict = process_one_dict(tokenizer, data[idx])
        res_list.append(one_res_dict)
    p.finish()
    print ('Finish processing {}'.format(file_name))
    save_file = output_path_prefix + r'/' + file_name
    with open(save_file, 'w') as outfile:
            json.dump(res_list, outfile, indent=4)

def process_source_prefix(path_prefix, tokenizer, output_path_prefix):
    file_name_list = os.listdir(path_prefix)
    for name in file_name_list:
        if name.endswith('.json'):
            pass
        else:
            continue
        process_file(path_prefix, name, tokenizer, output_path_prefix)

if __name__ == '__main__':
    save_path = r'../tokenized_pretraining_corpora/'
    tokenizer_path = save_path + r'/tokenizer_with_special_token'
    print ('Loading tokenizer...')
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print ('Tokenizer loaded.')
    
    source_path_prefix = r'../separate_datasets/'

    dataset_folder_name_list = ['CamRes676', 'Frames', 'KVRET', 'MetaLWOZ', 'MS_E2E', \
    'Schema_Guided', 'TaskMaster', 'WOZ']
    for dataset_name in dataset_folder_name_list:
        print ('Tokenizing {} Dataset...'.format(dataset_name))
        path_prefix = source_path_prefix + dataset_name + '/'
        process_source_prefix(path_prefix, tokenizer, save_path)
        print ('{} Dataset Tokenization Finished!'.format(dataset_name))




