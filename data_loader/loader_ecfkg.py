import os
import random
import collections

import torch
import numpy as np
import pandas as pd

from data_loader.loader_base import DataLoaderBase


class DataLoaderECFKG(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        self.relation_u2i_id = 0

        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        self.train_kg_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


