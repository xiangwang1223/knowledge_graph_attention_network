'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline model, NFM, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
from utility.load_data import Data
import scipy.sparse as sp

class NFM_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)
        # generate the sparse matrix for the knowledge graph features.
        kg_feat_file = path + '/kg_feat.npz'
        self.kg_feat_mat = self.get_kg_feature(kg_feat_file)

        # generate the one-hot sparse matrix for the users.
        self.user_one_hot = sp.identity(self.n_users).tocsr()

    def get_kg_feature(self, kg_feat_file):
        try:
            kg_feat_mat = sp.load_npz(kg_feat_file)
            print('already load item kg feature mat', kg_feat_mat.shape)
        except Exception:
            kg_feat_mat = self._create_kg_feat_mat()
            sp.save_npz(kg_feat_file, kg_feat_mat)
            print('already save item kg feature mat:', kg_feat_file)
        return kg_feat_mat

    def _create_kg_feat_mat(self):
        cat_rows = []
        cat_cols = []
        cat_data = []

        for i_id in range(self.n_items):
            # One-hot encoding for items.
            cat_rows.append(i_id)
            cat_cols.append(i_id)
            cat_data.append(1)

            # Multi-hot encoding for kg features of items.
            if i_id not in self.kg_dict.keys(): continue
            triples = self.kg_dict[i_id]
            for trip in triples:
                # ... only consider the tail entities.
                t_id = trip[0]
                # ... relations are ignored.
                r_id = trip[1]

                cat_rows.append(i_id)
                cat_cols.append(t_id)
                cat_data.append(1.)

        kg_feat_mat = sp.coo_matrix((cat_data, (cat_rows, cat_cols)), shape=(self.n_items, self.n_entities)).tocsr()
        return kg_feat_mat

    def generate_train_batch(self):

        users, pos_items, neg_items = self._generate_train_cf_batch()
        u_sp = self.user_one_hot[users]
        pos_i_sp = self.kg_feat_mat[pos_items]
        neg_i_sp = self.kg_feat_mat[neg_items]


        # Horizontally stack sparse matrices to get single positive & negative feature matrices
        pos_feats = sp.hstack([u_sp, pos_i_sp])
        neg_feats = sp.hstack([u_sp, neg_i_sp])

        batch_data = {}
        batch_data['pos_feats'] = pos_feats
        batch_data['neg_feats'] = neg_feats
        return batch_data

    def _extract_sp_info(self, sp_feats):
        sp_indices = np.hstack((sp_feats.nonzero()[0][:, None],
                                sp_feats.nonzero()[1][:, None]))
        sp_values = sp_feats.data
        sp_shape = sp_feats.shape
        return sp_indices, sp_values, sp_shape

    def generate_train_feed_dict(self, model, batch_data):

        pos_indices, pos_values, pos_shape = self._extract_sp_info(batch_data['pos_feats'])
        neg_indices, neg_values, neg_shape = self._extract_sp_info(batch_data['neg_feats'])

        feed_dict = {
            model.pos_indices:  pos_indices,
            model.pos_values: pos_values,
            model.pos_shape: pos_shape,

            model.neg_indices: neg_indices,
            model.neg_values: neg_values,
            model.neg_shape: neg_shape,

            model.mess_dropout: eval(self.args.mess_dropout)
        }

        return feed_dict

    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):
        user_list = np.repeat(user_batch, len(item_batch)).tolist()
        item_list = list(item_batch) * len(user_batch)

        u_sp = self.user_one_hot[user_list]
        pos_i_sp = self.kg_feat_mat[item_list]

        # Horizontally stack sparse matrices to get single positive & negative feature matrices
        pos_feats = sp.hstack([u_sp, pos_i_sp])
        pos_indices, pos_values, pos_shape = self._extract_sp_info(pos_feats)

        feed_dict = {
            model.pos_indices: pos_indices,
            model.pos_values: pos_values,
            model.pos_shape: pos_shape,

            model.mess_dropout: [0.] * len(eval(self.args.layer_size))
        }

        return feed_dict



