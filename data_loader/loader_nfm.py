import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderNFM(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)


    def construct_data(self, kg_data):
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        # construct feature matrix
        feat_rows = list(range(self.n_items))
        feat_cols = list(range(self.n_items))
        feat_data = [1] * self.n_items

        filtered_kg_data = kg_data[kg_data['h'] < self.n_items]
        feat_rows += filtered_kg_data['h'].tolist()
        feat_cols += filtered_kg_data['t'].tolist()
        feat_data += [1] * filtered_kg_data.shape[0]

        self.user_matrix = sp.identity(self.n_users).tocsr()
        self.feat_matrix = sp.coo_matrix((feat_data, (feat_rows, feat_cols)), shape=(self.n_items, self.n_entities)).tocsr()


    def print_info(self, logging):
        logging.info('n_users:              %d' % self.n_users)
        logging.info('n_items:              %d' % self.n_items)
        logging.info('n_entities:           %d' % self.n_entities)
        logging.info('n_users_entities:     %d' % self.n_users_entities)

        logging.info('n_cf_train:           %d' % self.n_cf_train)
        logging.info('n_cf_test:            %d' % self.n_cf_test)

        logging.info('shape of user_matrix: {}'.format(self.user_matrix.shape))
        logging.info('shape of feat_matrix: {}'.format(self.feat_matrix.shape))


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def generate_train_batch(self, user_dict):
        batch_user, batch_pos_item, batch_neg_item = self.generate_cf_batch(user_dict, self.train_batch_size)
        batch_user_sp = self.user_matrix[batch_user.numpy()]
        batch_pos_item_sp = self.feat_matrix[batch_pos_item.numpy()]
        batch_neg_item_sp = self.feat_matrix[batch_neg_item.numpy()]

        pos_feature_values = sp.hstack([batch_user_sp, batch_pos_item_sp])
        neg_feature_values = sp.hstack([batch_user_sp, batch_neg_item_sp])

        pos_feature_values = self.convert_coo2tensor(pos_feature_values.tocoo())
        neg_feature_values = self.convert_coo2tensor(neg_feature_values.tocoo())
        return pos_feature_values, neg_feature_values


    def generate_test_batch(self, batch_user):
        n_rows = len(batch_user) * self.n_items
        user_rows = list(range(n_rows))
        user_cols = np.repeat(batch_user, self.n_items)
        user_data = [1] * n_rows

        batch_user_sp = sp.coo_matrix((user_data, (user_rows, user_cols)), shape=(n_rows, self.n_users)).tocsr()
        batch_item_sp = sp.vstack([self.feat_matrix] * len(batch_user))

        feature_values = sp.hstack([batch_user_sp, batch_item_sp])
        feature_values = self.convert_coo2tensor(feature_values.tocoo())
        return feature_values


