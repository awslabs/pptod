domain_map_dict = {'Media_2': 'media',
 'Homes_2': 'homes',
 'Hotels_2': 'hotels',
 'Homes_1': 'homes',
 'Buses_2': 'buses',
 'RideSharing_2': 'ridesharing',
 'Services_3': 'services',
 'RentalCars_3': 'rentalcars',
 'Weather_1': 'weather',
 'Events_2': 'events',
 'Services_2': 'services',
 'Music_2': 'music',
 'Movies_3': 'movies',
 'Movies_1': 'movies',
 'Payment_1': 'payment',
 'Trains_1': 'trains',
 'Events_1': 'events',
 'Restaurants_2': 'restaurants',
 'Buses_3': 'buses',
 'Flights_2': 'flights',
 'Hotels_1': 'hotels',
 'Calendar_1': 'calendar',
 'Music_3': 'music',
 'Services_4': 'services',
 'Services_1': 'services',
 'RentalCars_2': 'rentalcars',
 'RentalCars_1': 'rentalcars',
 'Flights_1': 'flights',
 'Banks_1': 'banks',
 'Hotels_4': 'hotels',
 'RideSharing_1': 'ridesharing',
 'Restaurants_1': 'restaurants',
 'Events_3': 'events',
 'Travel_1': 'travel',
 'Media_3': 'media',
 'Music_1': 'music',
 'Messaging_1': 'messaging',
 'Hotels_3': 'hotels',
 'Buses_1': 'buses',
 'Movies_2': 'movies',
 'Flights_4': 'flights',
 'Alarm_1': 'alarm',
 'Media_1': 'media',
 'Banks_2': 'banks',
 'Flights_3': 'flights'}

action_map_dict = {
    'CONFIRM':'confirm',
    'GOODBYE':'goodbye',
    'INFORM':'inform',
    'INFORM_COUNT':'inform_count',
    'NOTIFY_FAILURE':'notify_failure',
    'NOTIFY_SUCCESS':'notify_success',
    'OFFER':'offer',
    'OFFER_INTENT':'offer_intent',
    'REQUEST':'request',
    'REQ_MORE':'reqmore'
}

import os
def list_file_names(path):
    file_list = os.listdir(path)
    res_list = []
    for file in file_list:
        if file.startswith('dialogue') and file.endswith('.json'):
            one_file = path + '/' + file
            res_list.append(one_file)
    return res_list

import json
def load_all_json_files(path):
    all_file_name_list = list_file_names(path)
    res_list = []
    for file in all_file_name_list:
        with open(file) as f:
            data = json.load(f)  
        res_list += data
    print (len(res_list))
    return res_list

def zip_turn_list(item):
    sess_list = item['turns']
    zip_turn_list = []
    one_turn_list = []
    target_speaker = "USER"
    target_map = {"USER":"SYSTEM",
                 "SYSTEM":"USER"}
    for turn in sess_list:
        if turn['speaker'] == target_speaker:
            target_speaker = target_map[turn['speaker']]
            one_turn_list.append(turn)
            if len(one_turn_list) == 2:
                zip_turn_list.append(one_turn_list)
                one_turn_list = []
        else:
            continue
    return zip_turn_list

def parse_text(text):
    token_list = text.split('_')
    return ' '.join(token_list).strip()

import re
def restore_text(text):
    text = re.sub('=', '', text)
    text = re.sub(',', '', text)
    text = ' '.join(text.split()).strip()
    return text

def transform_dict_to_text(one_domain_dict, domain):
    res_text = ''
    resdx_text = ''
    res_text += domain + ' '
    resdx_text += domain + ' '
    slot_name_list = one_domain_dict['slot_value_list']
    for name in slot_name_list:
        value = one_domain_dict[name]
        res_text += name + ' ' + value + ' '
        resdx_text += name + ' '
    res_text = res_text.strip().strip(',').strip()
    resdx_text = resdx_text.strip().strip(',').strip()
    return res_text, resdx_text
        
def extract_usr_belief_state(usr_dict, prev_bs_dict, prev_bs_name_list):
    res_bs_dict, res_bs_name_list = prev_bs_dict.copy(), prev_bs_name_list.copy()
    assert usr_dict['speaker'] == 'USER'
    frame_list = usr_dict['frames']
    res_text = ''
    for frame in frame_list:
        curr_domain = '[' + domain_map_dict[frame['service']] + ']'        
        slot_value_list = frame['state']["slot_values"]
        tmp_list = []
        for key in slot_value_list:
            slot_name = parse_text(key)
            slot_value = slot_value_list[key]
            if len(slot_value) == 0:
                continue
            else:
                tmp_list.append((slot_name, slot_value[0]))
                
        if len(tmp_list) == 0:
            continue
        else:
            # update domain dictionary
            try:
                res_bs_dict[curr_domain]
                assert curr_domain in res_bs_name_list
            except KeyError:
                res_bs_dict[curr_domain] = {'slot_value_list':[]}
                res_bs_name_list.append(curr_domain)
            for item in tmp_list:
                one_slot_name, one_slot_value = item
                if one_slot_name in res_bs_dict[curr_domain]['slot_value_list']:
                    pass
                else:
                    res_bs_dict[curr_domain]['slot_value_list'].append(one_slot_name)
                res_bs_dict[curr_domain][one_slot_name] = one_slot_value
    
    res_text = ''
    resdx_text = ''
    for domain in res_bs_name_list:
        one_domain_dict = res_bs_dict[domain]
        one_domain_text, one_domain_dx_text = transform_dict_to_text(one_domain_dict, domain)
        res_text += one_domain_text + ' '
        resdx_text += one_domain_dx_text + ' '
    return res_text.strip(), resdx_text.strip(), res_bs_dict, res_bs_name_list            

def get_one_domain_action_text(domain_action_dict, domain):
    res_text = domain + ' '
    action_type_list = domain_action_dict['action_type_list']
    for action_type in action_type_list:
        res_text += action_type + ' '
        value_text = ' '.join(domain_action_dict[action_type])
        res_text += value_text + ' '
    return res_text.strip()

def extract_system_action(system_dict):
    assert system_dict['speaker'] == "SYSTEM"
    frame_list = system_dict['frames']
    res_dict = {}
    action_domain_list = []
    for frame in frame_list:
        domain_name = '[' + domain_map_dict[frame["service"]] + ']'
        # first parse action list
        action_list = frame["actions"]
        if len(action_list) == 0: continue # no valid action
        try:
            res_dict[domain_name]
            assert domain_name in action_domain_list
        except KeyError:
            action_domain_list.append(domain_name)
            res_dict[domain_name] = {'action_type_list':[]}
        for act in action_list:
            action_type = '[' + action_map_dict[act["act"]] + ']'
            if action_type not in res_dict[domain_name]['action_type_list']:
                res_dict[domain_name]['action_type_list'].append(action_type)
            slot = ' '.join(act["slot"].split('_')).strip()
            try:
                res_dict[domain_name][action_type]
                assert action_type in res_dict[domain_name]['action_type_list']
            except KeyError:
                res_dict[domain_name][action_type] = []
            if slot in res_dict[domain_name][action_type]:
                pass
            else:
                res_dict[domain_name][action_type].append(slot)
    res_text = ''
    for domain in action_domain_list:
        one_domain_dict = res_dict[domain]
        one_domain_text = get_one_domain_action_text(one_domain_dict, domain)
        res_text += one_domain_text + ' '
    return ' '.join(res_text.split()).strip()

def zip_turn(one_turn_list, prev_bs_dict, prev_bs_name_list):
    usr_dict, system_dict = one_turn_list
    bs_text, bsdx_text, bs_dict, bs_name_list = extract_usr_belief_state(usr_dict, prev_bs_dict, prev_bs_name_list)
    action_text = extract_system_action(system_dict)
    usr_uttr = usr_dict["utterance"]
    system_uttr = system_dict["utterance"]
    return usr_uttr, bs_text, bsdx_text, system_uttr, action_text,  bs_dict, bs_name_list

def process_session_list(session_list):
    turn_num = len(session_list)
    if turn_num == 0: raise Exception()
    res_dict = {'dataset':'Schema_Guided',
               'dialogue_session':[]}
    for idx in range(turn_num):
        if idx == 0:
            bs_dict, bs_name_list = {}, []
        one_turn_list = session_list[idx]
        one_usr_uttr, one_usr_bs, one_usr_bsdx, one_system_uttr, one_system_action, \
        bs_dict, bs_name_list = zip_turn(one_turn_list, bs_dict, bs_name_list)
        one_turn_dict = {'turn_num':idx}
        one_turn_dict['user'] = one_usr_uttr
        one_turn_dict['resp'] = one_system_uttr
        one_turn_dict['turn_domain'] = bs_name_list
        one_turn_dict['bspn'] = one_usr_bs
        one_turn_dict['bsdx'] = one_usr_bsdx
        one_turn_dict['aspn'] = one_system_action
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

if __name__ == '__main__':
    print ('Processing Schema-Guided Dataset...')

    path = r'../raw_data/dstc8-schema-guided-dialogue/train/'
    train_json_data_list = load_all_json_files(path)

    train_list = []
    for item in train_json_data_list:
        one_turn_list = zip_turn_list(item)
        if len(one_turn_list) == 0: continue
        one_res_dict = process_session_list(one_turn_list)
        train_list.append(one_res_dict)
    print (len(train_list))

    path = r'../raw_data/dstc8-schema-guided-dialogue/dev/'
    dev_json_data_list = load_all_json_files(path)


    dev_list = []
    for item in dev_json_data_list:
        one_turn_list = zip_turn_list(item)
        if len(one_turn_list) == 0: continue
        one_res_dict = process_session_list(one_turn_list)
        dev_list.append(one_res_dict)
    print (len(dev_list))

    import os
    save_path = r'../separate_datasets/Schema_Guided/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)


    import json
    out_f = save_path + '/schema_guided_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(train_list, outfile, indent=4)

    out_f = save_path + r'/schema_guided_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(dev_list, outfile, indent=4)

    print ('Processing Schema-Guided Dataset Finished!')


