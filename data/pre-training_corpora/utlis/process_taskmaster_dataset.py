domain_term_dict = {}
domain_term_dict['uber_lyft'] = {'location.from':'from location',
                                'location.to':'to location',
                                'type.ride':'ride type',
                                'num.people':'number of people',
                                'price.estimate':'estimate price',
                                'duration.estimate':'estimate duration',
                                'time.pickup':'pickup time',
                                'time.dropoff':'dropoff time'}

domain_term_dict['movie_ticket'] = {'name.movie':'movie name',
                                   'name.theater':'theater name',
                                   'num.tickets':'number of tickets',
                                   'time.start':'start time',
                                   'location.theater':'theater location',
                                   'price.ticket':'ticket price',
                                   'type.screening':'screening type',
                                   'time.end':'end time',
                                   'time.duration':'duration time'}

domain_term_dict['restaurant_reservation'] = {'name.restaurant':'restaurant name',
                                             'name.reservation':'reservation name',
                                             'num.guests':'number of guests',
                                             'time.reservation':'reservation time',
                                             'type.seating':'seating type',
                                             'location.restaurant':'restaurant location'}

domain_term_dict['coffee_ordering'] = {'location.store':'store location',
                                      'name.drink':'drink name',
                                      'size.drink':'drink size',
                                      'num.drink':'number of drink',
                                      'type.milk':'milk type',
                                      'preference':'preference'}

domain_term_dict['pizza_ordering'] = {'name.store':'store name',
                                     'name.pizza':'pizza name',
                                     'size.pizza':'pizza size',
                                     'type.topping':'topping type',
                                     'type.crust':'crust type',
                                     'preference':'preference',
                                     'location.store':'store location'}

domain_term_dict['auto_repair'] = {'name.store':'store name',
                                  'name.customer':'customer name',
                                  'date.appt':'appt date',
                                  'time.appt':'appt time',
                                  'reason.appt':'appt reason',
                                  'name.vehicle':'vehicle name',
                                  'year.vehicle':'vehicle year',
                                  'location.store':'store location'}

domain_list = ['uber_lyft', 'movie_ticket', 'restaurant_reservation', 
              'coffee_ordering', 'pizza_ordering', 'auto_repair']

def identify_domain(text):
    for domain in domain_list:
        if domain in text:
            return domain
    return ''

def build_bs_text(bs_dict, bs_name_list, bs_domain):
    if len(bs_dict) == 0 or len(bs_name_list) == 0 or bs_domain == '':
        return '', '' # empty belief state text
    bs_domain = '[' + bs_domain + ']'
    bs_text = bs_domain + ' '
    bsdx_text = bs_domain + ' '
    for name in bs_name_list:
        bs_text += name + ' ' + bs_dict[name].strip() + ' '
        bsdx_text += name + ' '
    bs_text = ' '.join(bs_text.split()).strip(',').strip()
    bsdx_text = ' '.join(bsdx_text.split()).strip(',').strip()
    return bs_text, bsdx_text
        
def extract_one_uttr_bs(prev_bs_dict, prev_bs_list, prev_domain, usr_dict):
    res_bs_dict, res_bs_list, res_domain = prev_bs_dict.copy(), \
    prev_bs_list.copy(), prev_domain
    try:
        anno_list = usr_dict['segments']
        curr_bs_dict = {}
        for item in anno_list:
            value = item['text']
            slot = item['annotations'][0]['name']
            item_domain = identify_domain(slot)
            if res_domain == '': 
                res_domain = item_domain # update domain
            else:
                pass
            
            match_flag = False
            for key in domain_term_dict[item_domain]:
                if key in slot:
                    slot = domain_term_dict[item_domain][key]
                    match_flag = True
                    break
            if match_flag: # find valid slot
                curr_bs_dict[slot] = value.strip(',').strip()
        for key in curr_bs_dict:
            if key in res_bs_dict: 
                pass
            else:
                res_bs_list.append(key)
            res_bs_dict[key] = curr_bs_dict[key] # update belief state

        bs_text, bsdx_text = build_bs_text(res_bs_dict, res_bs_list, res_domain)
    except KeyError:
        bs_text, bsdx_text = build_bs_text(prev_bs_dict, prev_bs_list, prev_domain)
    return bs_text, bsdx_text, res_bs_dict, res_bs_list, res_domain

def zip_turn(usr_dict, system_dict, bs_dict, bs_list, bs_domain):
    assert usr_dict['speaker'] == 'USER'
    assert system_dict['speaker'] == 'ASSISTANT'
    usr_uttr = usr_dict['text']
    system_uttr = system_dict['text']
    bs_text, bsdx_text, bs_dict, bs_list, bs_domain = extract_one_uttr_bs(bs_dict, bs_list, bs_domain, usr_dict)
    return usr_uttr, bs_text, bsdx_text, system_uttr, bs_dict, bs_list, bs_domain

def build_session_list(in_item):
    raw_session_list = in_item['utterances']
    zip_turn_list = []
    one_turn_list = []
    target_speaker = 'USER'
    target_map = {'USER':'ASSISTANT',
                 'ASSISTANT':'USER'}
    for sess in raw_session_list:
        if sess['speaker'] == target_speaker:
            target_speaker = target_map[sess['speaker']]
            one_turn_list.append(sess)
            if len(one_turn_list) == 2:
                zip_turn_list.append(one_turn_list)
                one_turn_list = []
        else:
            continue
    return zip_turn_list

def process_session(sess_list):
    res_dict = {'dataset': 'TaskMaster',
               'dialogue_session':[]}
    turn_num = len(sess_list)
    for idx in range(turn_num):
        one_turn_dict = {'turn_num': idx}
        if idx == 0:
            bs_dict, bs_list, bs_domain = {}, [], ''
        usr_uttr, bs_text, bsdx_text, system_uttr, bs_dict, bs_list, bs_domain = \
        zip_turn(sess_list[idx][0], sess_list[idx][1], bs_dict, bs_list, bs_domain)
        one_turn_dict['user'] = usr_uttr
        one_turn_dict['resp'] = system_uttr
        one_turn_dict['turn_domain'] = [bs_domain]
        one_turn_dict['bspn'] = bs_text
        one_turn_dict['bsdx'] = bsdx_text
        one_turn_dict['aspn'] = ''
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

import json
def process_file(in_f):
    with open(in_f) as f:
        data = json.load(f)   
    
    all_session_list = []
    for item in data:
        one_session_list = build_session_list(item)
        if len(one_session_list) > 0:
            all_session_list.append(one_session_list)
        else:
            pass
    res_list = []
    for sess in all_session_list:
        one_res_dict = process_session(sess)
        res_list.append(one_res_dict)
    print (len(res_list), len(data))
    return res_list

if __name__ == '__main__':
    print ('Processing TaskMaster Dataset...')
    in_f = r'../raw_data/Taskmaster/TM-1-2019/self-dialogs.json'
    self_dialogue_list = process_file(in_f)

    in_f = r'../raw_data/Taskmaster/TM-1-2019/woz-dialogs.json'
    woz_dialogue_list = process_file(in_f)

    all_data_list = self_dialogue_list + woz_dialogue_list
    print (len(all_data_list))

    import random
    random.shuffle(all_data_list)

    dev_data_list = all_data_list[:1000]
    train_data_list = all_data_list[1000:]

    import os
    save_path = r'../separate_datasets/TaskMaster/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    out_f = save_path + r'/taskmaster_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(dev_data_list, outfile, indent=4)

    out_f = save_path + r'/taskmaster_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(train_data_list, outfile, indent=4)
    print ('Processing TaskMaster Dataset Finished!')








