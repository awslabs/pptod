def update_belief_state(usr_dict, prev_bs_dict, prev_bs_name_list):
    res_bs_dict, res_bs_name_list = prev_bs_dict, prev_bs_name_list
    curr_bs_state = usr_dict['belief_state']
    for item in curr_bs_state:
        if item['act'] == 'inform': # only care about inform act
            for pair in item['slots']:
                slot_name, value = pair
                if slot_name not in res_bs_name_list:
                    res_bs_name_list.append(slot_name)
                res_bs_dict[slot_name] = value
    if len(res_bs_name_list) == 0:
        res_text, res_dx_text = '', ''
    else:
        res_text = '[restaurant] '
        res_dx_text = '[restaurant] '
        for name in res_bs_name_list:
            value = res_bs_dict[name]
            res_text += name + ' ' + value + ' '
            res_dx_text += name + ' '
        res_text = res_text.strip().strip(' , ').strip()
        res_dx_text = res_dx_text.strip().strip(' , ').strip()
    return res_text, res_dx_text, res_bs_dict, res_bs_name_list

def zip_sess_list(sess_list):
    turn_num = len(sess_list)
    assert sess_list[0]["system_transcript"] == ''
    if turn_num == 1:
        raise Exception()
    turn_list = []
    for idx in range(turn_num - 1):
        curr_turn_dict = sess_list[idx]
        system_uttr = sess_list[idx+1]['system_transcript']
        turn_list.append((curr_turn_dict, system_uttr))
    return turn_list

def process_session(sess_list):
    turn_num = len(sess_list)
    res_dict = {'dataset':'WOZ',
               'dialogue_session':[]}
    for idx in range(turn_num):
        if idx == 0:
            bs_dict, bs_name_list = {}, []
        one_usr_dict, one_system_uttr = sess_list[idx]
        one_usr_uttr = one_usr_dict['transcript']
        one_usr_bs, one_usr_bsdx, bs_dict, bs_name_list = \
        update_belief_state(one_usr_dict, bs_dict, bs_name_list)
        
        one_turn_dict = {'turn_num':idx}
        one_turn_dict['user'] = one_usr_uttr
        one_turn_dict['resp'] = one_system_uttr
        one_turn_dict['turn_domain'] = ['[restaurant]']
        one_turn_dict['bspn'] = one_usr_bs
        one_turn_dict['bsdx'] = one_usr_bsdx
        one_turn_dict['aspn'] = ''
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

import json
def process_file(in_f):
    with open(in_f) as f:
        data = json.load(f) 
    res_list = []
    for item in data:
        one_sess = zip_sess_list(item['dialogue'])
        if len(one_sess) == 0:
            continue
        one_res_dict = process_session(one_sess)
        res_list.append(one_res_dict)
    print (len(res_list), len(data))
    return res_list

if __name__ == '__main__':
    print ('Processing WOZ Dataset...')
    in_f = r'../raw_data/neural-belief-tracker/data/woz/woz_train_en.json'
    train_list = process_file(in_f)

    in_f = r'../raw_data/neural-belief-tracker/data/woz/woz_validate_en.json'
    dev_list = process_file(in_f)

    in_f = r'../raw_data/neural-belief-tracker/data/woz/woz_test_en.json'
    test_list = process_file(in_f)

    all_data_list = train_list + test_list

    import os
    save_path = r'../separate_datasets/WOZ/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    import json
    out_f = save_path + r'/woz_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(all_data_list, outfile, indent=4)

    out_f = save_path + r'/woz_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(dev_list, outfile, indent=4)

    print ('Processing WOZ Dataset Finished!')
