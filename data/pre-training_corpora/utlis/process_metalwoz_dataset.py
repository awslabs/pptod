def zip_turn(usr_uttr, system_uttr, domain, turn_num):
    one_turn_dict = {'turn_num': turn_num}
    one_turn_dict['user'] = usr_uttr
    one_turn_dict['resp'] = system_uttr
    one_turn_dict['turn_domain'] = [domain]
    one_turn_dict['bspn'] = ''
    one_turn_dict['bsdx'] = ''
    one_turn_dict['aspn'] = ''
    return one_turn_dict

import json
def parse_one_instance(line):
    res_dict = {'dataset':'MetalWOZ',
                       'dialogue_session':[]}
    
    one_dict = json.loads(line.strip('\n'))
    dialogue_history = one_dict['turns'][1:] # the first sentence is always 'Hello how may I help you?'
    try:
        assert len(dialogue_history) % 2 == 0
    except:
        dialogue_history = dialogue_history[:-1]
    turn_num = int(len(dialogue_history) / 2)
    domain = '[' + one_dict['domain'].lower() + ']'
    for idx in range(turn_num):
        start_idx, end_idx = idx*2, idx*2 + 1
        usr_uttr, system_uttr = dialogue_history[start_idx], dialogue_history[end_idx]
        one_turn_dict = zip_turn(usr_uttr, system_uttr, domain, turn_num=idx)
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

import json
def process_file(in_f):
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        
    res_list = []
    all_num = 0
    valid_turn_num, max_turn_num = 0, 0
    for line in lines:
        all_num += 1
        sess_dict = parse_one_instance(line)
        if len(sess_dict['dialogue_session']) == 0: continue
        res_list.append(sess_dict)
        valid_turn_num += len(sess_dict['dialogue_session'])
        max_turn_num = max(max_turn_num, len(sess_dict['dialogue_session']))
    ave_turn_num = valid_turn_num / len(res_list)
    return res_list, all_num, len(res_list), ave_turn_num, max_turn_num

if __name__ == '__main__':
    print ('Processing MetalWOZ Dataset...')
    import os
    root_path = r'../raw_data/metalwoz/dialogues/'
    file_list = os.listdir(root_path)
    all_res_list = []
    all_session_num, all_turn_num = 0, 0
    for file in file_list:
        if file.endswith('.txt'):
            one_res_list, _, one_valid_session_num, one_ave_turn_num, _ = process_file(root_path + file)
            all_turn_num += one_valid_session_num * one_ave_turn_num
            all_session_num += one_valid_session_num
            all_res_list += one_res_list

    print (len(all_res_list))
    domain_set = set()
    for item in all_res_list:
        domain_set.add(item['dialogue_session'][0]['turn_domain'][0])
    print (len(domain_set))

    idx_list = [idx for idx in range(len(all_res_list))]
    import random
    random.shuffle(idx_list)

    dev_idx_set = set(idx_list[:3500])
    train_idx_set = set(idx_list[3500:])

    metalwoz_train_list, metalwoz_dev_list = [], []
    for idx in range(len(all_res_list)):
        if idx in train_idx_set:
            metalwoz_train_list.append(all_res_list[idx])
        else:
            metalwoz_dev_list.append(all_res_list[idx])
    print (len(metalwoz_train_list), len(metalwoz_dev_list))

    import random
    import json
    import os
    save_path = r'../separate_datasets/MetaLWOZ/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    out_f = save_path + r'/metalwoz_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(metalwoz_train_list, outfile, indent=4)
    
    out_f = save_path + r'/metalwoz_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(metalwoz_dev_list, outfile, indent=4)
    print ('Processing MetalWOZ Dataset Finished!')
    
