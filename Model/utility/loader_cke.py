'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, CKE, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
from utility.load_data import Data

class CKE_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)

    def _generate_train_kg_batch(self):
        exist_heads = self.kg_dict.keys()
        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            # pos triples associated with head entity h.
            # format of kg_dict is {h: [t,r]}.
            pos_triples = self.kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break

                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_entities, size=1)[0]

                if (t, r) not in self.kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        users, pos_items, neg_items = self._generate_train_cf_batch()
        heads, relations, pos_tails, neg_tails = self._generate_train_kg_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict ={
            model.u: batch_data['users'],
            model.pos_i: batch_data['pos_items'],
            model.neg_i: batch_data['neg_items'],

            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails']
        }

        return feed_dict

    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=False):
        feed_dict = {
            model.u: user_batch,
            model.pos_i: item_batch
        }
        return feed_dict