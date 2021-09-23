token_map = {'address':'address',
             'agenda':'agenda',
             'date':'date',
             'distance':'distance',
             'event':'event',
             'location':'location',
             'party':'party',
             'poi':'point of interest',
             'poi_type':'point of interest type',
             'room':'room',
             'time':'time',
             'traffic_info':'traffic information',
             'weather_attribute':'weather attribute'}

def update_belief_state(prev_bs_dict, prev_slot_name_list, curr_slot_dict):
    '''
        prev_bs_dict: a dictionary stores the slot and value pair for belief state of previous turns
        prev_slot_name_list: this list specifies the order of slot-value pair appears in the resulting belief state
        curr_slot_dict: the dictionary contains the slot-value pair for current turn
    '''
    res_bs_dict = prev_bs_dict.copy()
    res_slot_name_list = prev_slot_name_list.copy()
    for slot in curr_slot_dict:
        if slot in res_bs_dict: # check if the slot exists
            res_bs_dict[slot] = curr_slot_dict[slot] # update the slot value
        else:
            res_slot_name_list.append(slot)
            res_bs_dict[slot] = curr_slot_dict[slot]
    res_list = []
    for name in res_slot_name_list:
        res_list.append((name, res_bs_dict[name]))
    return res_list, res_bs_dict, res_slot_name_list

domain = r'[car_assistant]'
def zip_turn(turn_list, prev_bs_dict, prev_slot_name_list):
    assert turn_list[0]['turn'] == 'driver'
    assert turn_list[1]['turn'] == 'assistant'
    user_item, assistant_item = turn_list[0], turn_list[1]
    user_utterance = user_item['data']['utterance']
    system_utterance = assistant_item['data']['utterance']
    curr_slot_dict = assistant_item['data']['slots']
    res_bs_list, res_bs_dict, res_slot_name_list = update_belief_state(prev_bs_dict, 
                                                          prev_slot_name_list, curr_slot_dict)
    return user_utterance, system_utterance, res_bs_list, res_bs_dict, res_slot_name_list

def get_bs_text(bs_list):
    if len(bs_list) == 0: # no belief state
        return '', ''
    bs_text, bsdx_text = domain + ' ', domain + ' '
    for item in bs_list:
        slot = token_map[item[0]]
        bs_text += slot + ' ' + item[1] + ' '
        bsdx_text += slot + ' '
    bs_text = ' '.join(bs_text.strip().strip(',').strip().split())
    bsdx_text = ' '.join(bsdx_text.strip().strip(',').strip().split())
    return bs_text, bsdx_text

def build_session_list(item):
    zip_turn_list = []
    one_turn_list = []
    target_speaker = 'driver'
    target_map = {'driver':'assistant',
                 'assistant':'driver'}
    for sess in item:
        if sess['turn'] == target_speaker:
            target_speaker = target_map[sess['turn']]
            one_turn_list.append(sess)
            if len(one_turn_list) == 2:
                zip_turn_list.append(one_turn_list)
                one_turn_list = []
        else:
            continue
    return zip_turn_list

def process_dialogue_session(session_list):
    turn_num = len(session_list)
    res_dict = {'dataset':'KVRET',
               'dialogue_session':[]}
    for idx in range(turn_num):
        if idx == 0:
            bs_dict, slot_name_list = {}, []
        one_turn_list = session_list[idx]
        
        user_utterance, system_utterance, bs_list, bs_dict, slot_name_list = \
        zip_turn(one_turn_list, bs_dict, slot_name_list)
        
        bs_text, bsdx_text = get_bs_text(bs_list)
        
        one_turn_dict = {'turn_num':idx}
        one_turn_dict['user'] = user_utterance
        one_turn_dict['resp'] = system_utterance
        one_turn_dict['turn_domain'] = [domain]
        one_turn_dict['bspn'] = bs_text
        one_turn_dict['bsdx'] = bsdx_text
        one_turn_dict['aspn'] = ''
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

import json
def process_file(in_f):
    with open(in_f) as f:
        data = json.load(f)
        
    res_list = []
    for item in data:
        one_sess_list = build_session_list(item['dialogue'])
        if len(one_sess_list) == 0: continue
        one_res_dict = process_dialogue_session(one_sess_list)
        res_list.append(one_res_dict)
    print (len(res_list), len(data))
    return res_list

if __name__ == '__main__':
    print ('Processing KVRET Dataset...')
    import random
    import json
    import os
    save_path = r'../separate_datasets/KVRET/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    in_f = r'../raw_data/kvret/kvret_train_public.json'
    train_data_list = process_file(in_f)
    in_f = r'../raw_data/kvret/kvret_dev_public.json'
    dev_data_list = process_file(in_f)
    data_list = train_data_list + dev_data_list

    out_f = save_path + r'/kvret_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(data_list, outfile, indent=4)

    in_f = r'../raw_data/kvret/kvret_test_public.json'
    test_data_list = process_file(in_f)
    out_f = save_path + r'/kvret_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(test_data_list, outfile, indent=4)
    print ('Processing KVRET Dataset Finished!')


