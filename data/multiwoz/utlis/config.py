import logging, time, os

class Config:
    def __init__(self, data_prefix):
        # data_prefix = r'../data/'
        self.data_prefix = data_prefix
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        self.vocab_path_train = self.data_prefix + '/multi-woz-processed/vocab'
        self.data_path = self.data_prefix + '/multi-woz-processed/'
        self.data_file = 'data_for_damd.json'
        self.dev_list = self.data_prefix + '/multi-woz/valListFile.json'
        self.test_list = self.data_prefix + '/multi-woz/testListFile.json'

        self.dbs = {
            'attraction': self.data_prefix + '/db/attraction_db_processed.json',
            'hospital': self.data_prefix + '/db/hospital_db_processed.json',
            'hotel': self.data_prefix + '/db/hotel_db_processed.json',
            'police': self.data_prefix + '/db/police_db_processed.json',
            'restaurant': self.data_prefix + '/db/restaurant_db_processed.json',
            'taxi': self.data_prefix + '/db/taxi_db_processed.json',
            'train': self.data_prefix + '/db/train_db_processed.json',
        }
        self.domain_file_path = self.data_prefix + '/multi-woz-processed/domain_files.json'
        self.slot_value_set_path = self.data_prefix + '/db/value_set_processed.json'

        '''
        self.vocab_path_train = '../data/multi-woz-processed/vocab'
        self.data_path = '../data/multi-woz-processed/'
        self.data_file = 'data_for_damd.json'
        self.dev_list = '../data/multi-woz/valListFile.json'
        self.test_list = '../data/multi-woz/testListFile.json'



        self.dbs = {
            'attraction': '../data/db/attraction_db_processed.json',
            'hospital': '../data/db/hospital_db_processed.json',
            'hotel': '../data/db/hotel_db_processed.json',
            'police': '../data/db/police_db_processed.json',
            'restaurant': '../data/db/restaurant_db_processed.json',
            'taxi': '../data/db/taxi_db_processed.json',
            'train': '../data/db/train_db_processed.json',
        }
        self.domain_file_path = '../data/multi-woz-processed/domain_files.json'
        self.slot_value_set_path = '../data/db/value_set_processed.json'
        '''

        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi

        self.enable_aspn = True
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False

        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi
        self.max_context_length = 900
        self.vocab_size = 3000

