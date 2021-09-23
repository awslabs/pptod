token_map_dict = {'car_type':'car type',
                 'critic_rating': 'critic rating',
                 'distanceconstraints':'distance constraints',
                 'dress_code': 'dress code',
                 'dropoff_location':'dropoff location',
                 'dropoff_location_city':'dropoff location city',
                 'implicit_value':'implicit value',
                 'mc_list':'mc list',
                 'mealtype':'meal type',
                 'movie_series':'movie series',
                 'moviename':'movie name',
                 'mpaa_rating':'mpaa rating',
                 'multiple_choice':'multiple choice',
                 'numberofkids':'number of kids',
                 'numberofpeople':'number of people',
                 'personfullname':'person full name',
                 'phonenumber':'phone number',
                 'pickup_location':'pickup location',
                 'pickup_location_city':'pickup location city',
                 'pickup_time':'pickup time',
                 'restaurantname':'restaurant name',
                 'restauranttype':'restaurant type',
                 'starttime':'start time',
                 'taskcomplete': 'task complete',
                 'taskcomplete)':'task complete',
                 'theater_chain':'theater chain',
                 'video_format':'video format'}

def split_session_list(in_f):
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()[1:]
    curr_track_id = 1
    all_session_list = []
    one_session_list = []
    for line in lines:
        item_list = line.strip('\n').split('\t')
        curr_sess_id = int(item_list[0])
        if curr_sess_id != curr_track_id:
            all_session_list.append(one_session_list)
            one_session_list = [line.strip('\n')]
            curr_track_id = curr_sess_id # update track id
        else:
            one_session_list.append(line.strip('\n'))
    if len(one_session_list) > 0:
        all_session_list.append(one_session_list)
    return all_session_list

def build_session_list(sess_text_list):
    zip_turn_list = []
    one_turn_list = []
    target_speaker = 'user'
    target_map = {'user':'agent',
                 'agent':'user'}
    for sess_text in sess_text_list:
        if sess_text.strip('\n').split('\t')[3] == target_speaker:
            target_speaker = target_map[sess_text.strip('\n').split('\t')[3]]
            one_turn_list.append(sess_text)
            if len(one_turn_list) == 2:
                zip_turn_list.append(one_turn_list)
                one_turn_list = []
        else:
            continue
    return zip_turn_list

def parse_usr_goal(text):
    item_list = text.split('(')
    assert len(item_list) >= 2
    tuple_list = item_list[1].strip().strip(')').split(';')
    bs_list = []
    for one_tuple in tuple_list:
        one_tuple_split = one_tuple.split('=')
        if len(one_tuple_split) == 1:
            continue
        bs_list.append((one_tuple_split[0].strip(), one_tuple_split[1].strip()))
    return bs_list

def update_belief_state(prev_bs_list, text):
    res_list = prev_bs_list.copy()
    prev_slot_set = set()
    for item in res_list:
        prev_slot_set.add(item[0])
    
    res_slot_set = prev_slot_set.copy()
    curr_bs_list = parse_usr_goal(text)
    for item in curr_bs_list:
        if item[0] in prev_slot_set: continue
        res_list.append(item)
    return res_list

def update_belief_state(prev_bs_dict, prev_bs_name_list, text):
    res_bs_dict = prev_bs_dict.copy()
    res_bs_name_list = prev_bs_name_list.copy()
    try:
        curr_bs_list = parse_usr_goal(text)
    except AssertionError:
        #print (text)
        raise Exception()
    for item in curr_bs_list:
        slot, value = item
        if slot in res_bs_dict: 
            res_bs_dict[slot] = value # update value
        else:
            res_bs_name_list.append(slot)
            res_bs_dict[slot] = value # add new value
    return res_bs_dict, res_bs_name_list
    

def parse_usr_belief_state(prev_bs_dict, prev_bs_name_list, text, domain):
    split_list = text.split('\t')[5:]
    #bs_dict, bs_name_list = {}, []
    bs_dict, bs_name_list = prev_bs_dict.copy(), prev_bs_name_list.copy()
    for text in split_list:
        #print (text)
        if len(text) == 0:
            break
        bs_dict, bs_name_list = update_belief_state(bs_dict, bs_name_list, text)
    bs_text = ''
    bsdx_text = ''
    for name in bs_name_list:
        try:
            slot = token_map_dict[name]
        except KeyError:
            slot = name
        bs_text += slot + ' ' + bs_dict[name] + ' '
        bsdx_text += slot + ' '
    bs_text = bs_text.strip().strip(',').strip()
    bsdx_text = bsdx_text.strip().strip(',').strip()
    if len(bs_text) == 0:
        bs_text = ''
    else:
        bs_text = domain + ' ' + bs_text
    if len(bsdx_text) == 0:
        bsdx_text = ''
    else:
        bsdx_text = domain + ' ' + bsdx_text.strip()
    return ' '.join(bs_text.split()).strip(), ' '.join(bsdx_text.split()).strip(), bs_dict, bs_name_list

def parse_one_agent_action(text):
    item_list = text.split('(')
    assert len(item_list) >= 2
    action_type = '[' + item_list[0].strip() + ']'
    action_text = action_type + ' '
    tuple_list = item_list[1].strip().strip(')').split(';')
    action_list = []
    for one_tuple in tuple_list:
        one_action = one_tuple.split('=')[0].strip()
        try:
            one_action = token_map_dict[one_action]
        except KeyError:
            one_action = one_action
        action_list.append(one_action)
    return action_type, action_list

def parse_agent_action(text, domain):
    split_list = text.split('\t')[5:]
    res_list = []
    action_dict, action_type_list = {}, []
    for text in split_list:
        if len(text) == 0:
            break
        else:
            one_action_type, one_action_list = parse_one_agent_action(text)
            try:
                for a in one_action_list:
                    if a in action_dict[one_action_type]:
                        pass
                    else:
                        action_dict[one_action_type].append(a)
            except KeyError:
                action_type_list.append(one_action_type)
                action_dict[one_action_type] = one_action_list
    res_text = domain + ' '
    for key in action_type_list:
        res_text += key + ' '
        one_list = action_dict[key]
        for item in one_list:
            res_text += item + ' '
        res_text = res_text.strip().strip(',').strip() + ' '
    return ' '.join(res_text.split()).strip()
    
def zip_turn(prev_bs_dict, prev_bs_name_list, turn_list, domain):
    usr_text, agent_text = turn_list
    try:
        assert usr_text.strip('\n').split('\t')[3] == 'user'
        assert agent_text.strip('\n').split('\t')[3] == 'agent'
    except:
        raise Exception()
    usr_uttr = usr_text.strip('\n').split('\t')[4].strip()
    usr_bs, usr_bsdx, bs_dict, bs_name_list = \
    parse_usr_belief_state(prev_bs_dict, prev_bs_name_list, usr_text, domain)
    system_uttr = agent_text.strip('\n').split('\t')[4].strip()
    system_action = parse_agent_action(agent_text, domain)
    return usr_uttr, usr_bs, usr_bsdx, system_uttr, system_action, bs_dict, bs_name_list

def process_session_list(session_list, domain):
    turn_num = len(session_list)
    if turn_num == 0: raise Exception()
    res_dict = {'dataset':'E2E_MS',
               'dialogue_session':[]}
    for idx in range(turn_num):
        if idx == 0:
            bs_dict, bs_name_list = {}, []
        one_turn_list = session_list[idx]
        one_usr_uttr, one_usr_bs, one_usr_bsdx, one_system_uttr, one_system_action, \
        bs_dict, bs_name_list = zip_turn(bs_dict, bs_name_list, one_turn_list, domain)
        
        one_turn_dict = {'turn_num':idx}
        one_turn_dict['user'] = one_usr_uttr
        one_turn_dict['resp'] = one_system_uttr
        one_turn_dict['turn_domain'] = [domain]
        one_turn_dict['bspn'] = one_usr_bs
        one_turn_dict['bsdx'] = one_usr_bsdx
        one_turn_dict['aspn'] = one_system_action
        res_dict['dialogue_session'].append(one_turn_dict)
    return res_dict

def process_file(in_f, domain):
    all_session_list = split_session_list(in_f)
    res_list = []
    for item in all_session_list:
        one_sess = build_session_list(item)
        if len(one_sess) == 0:
            continue
        one_res_dict = process_session_list(one_sess, domain)
        res_list.append(one_res_dict)
    print (len(res_list), len(all_session_list))
    return res_list

if __name__ == '__main__':
    print ('Processing MSE2E Dataset...')
    in_f = r'../raw_data/e2e_dialog_challenge/data/taxi_all.tsv'
    domain = '[taxi]'
    taxi_res_list = process_file(in_f, domain)

    in_f = r'../raw_data/e2e_dialog_challenge/data/movie_all.tsv'
    domain = '[movie]'
    movie_res_list = process_file(in_f, domain)

    in_f = r'../raw_data/e2e_dialog_challenge/data/restaurant_all.tsv'
    domain = '[restaurant]'
    restaurant_res_list = process_file(in_f, domain)

    all_data_list = taxi_res_list + movie_res_list + restaurant_res_list
    len(all_data_list)

    import random
    random.shuffle(all_data_list)
    test_data_list = all_data_list[:500]
    train_data_list = all_data_list[500:]

    import random
    import json
    import os
    save_path = r'../separate_datasets/MS_E2E/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    import json
    out_f = save_path + r'/e2e_ms_train.json'
    with open(out_f, 'w') as outfile:
        json.dump(train_data_list, outfile, indent=4)

    out_f = save_path + r'/e2e_ms_test.json'
    with open(out_f, 'w') as outfile:
        json.dump(test_data_list, outfile, indent=4)
    print ('Processing MSE2E Dataset Finished!')
    