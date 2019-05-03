'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, CKE, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class CKE(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model()
        self._build_loss()
        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = 'cke'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.lr = args.lr
        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size

        self.regs = eval(args.regs)

        self.verbose = args.verbose

    def _build_inputs(self):
        # for user-item interaction modelling
        self.u = tf.placeholder(tf.int32, shape=[None,], name='u')
        self.pos_i = tf.placeholder(tf.int32, shape=[None,], name='pos_i')
        self.neg_i = tf.placeholder(tf.int32, shape=[None,], name='neg_i')

        # for knowledge graph modeling (TransD)
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')


    def _build_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['item_embed'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embed')
            print('using xavier initialization')
        else:
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)
            all_weights['item_embed'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                    name='item_embed', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['kg_entity_embed'] = tf.Variable(initializer([self.n_entities, 1, self.emb_dim]),
                                                     name='kg_entity_embed')
        all_weights['kg_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                       name='kg_relation_embed')

        all_weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]))

        return all_weights

    def _build_model(self):
        self.u_e, self.pos_i_e, self.neg_i_e = self._get_cf_inference(self.u, self.pos_i, self.neg_i)

        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e= self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)

        # All predictions for all users.
        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)


    def _build_loss(self):
        self.kg_loss, self.kg_reg_loss = self._get_kg_loss()
        self.cf_loss, self.cf_reg_loss = self._get_cf_loss()

        self.base_loss = self.cf_loss
        self.kge_loss = self.kg_loss
        self.reg_loss = self.regs[0] * self.cf_reg_loss + self.regs[1] * self.kg_reg_loss
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        # head & tail entity embeddings: batch_size *1 * emb_dim
        h_e = tf.nn.embedding_lookup(self.weights['kg_entity_embed'], h)
        pos_t_e = tf.nn.embedding_lookup(self.weights['kg_entity_embed'], pos_t)
        neg_t_e = tf.nn.embedding_lookup(self.weights['kg_entity_embed'], neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.weights['kg_relation_embed'], r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.kge_dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.kge_dim])

        # l2-normalize
        h_e = tf.math.l2_normalize(h_e, axis=1)
        r_e = tf.math.l2_normalize(r_e, axis=1)
        pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _get_cf_inference(self, u, pos_i, neg_i):
        u_e = tf.nn.embedding_lookup(self.weights['user_embed'], u)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embed'], pos_i)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embed'], neg_i)

        pos_i_kg_e = tf.reshape(tf.nn.embedding_lookup(self.weights['kg_entity_embed'], pos_i), [-1, self.emb_dim])
        neg_i_kg_e = tf.reshape(tf.nn.embedding_lookup(self.weights['kg_entity_embed'], neg_i), [-1, self.emb_dim])

        return u_e, pos_i_e + pos_i_kg_e, neg_i_e + neg_i_kg_e

    def _get_kg_loss(self):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)

        maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        kg_loss = tf.negative(tf.reduce_mean(maxi))
        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)

        return kg_loss, kg_reg_loss

    def _get_cf_loss(self):
        def _get_cf_score(u_e, i_e):
            cf_score = tf.reduce_sum(tf.multiply(u_e, i_e), axis=1)
            return cf_score

        pos_cf_score = _get_cf_score(self.u_e, self.pos_i_e)
        neg_cf_score = _get_cf_score(self.u_e, self.neg_i_e)

        maxi = tf.log(1e-10 + tf.nn.sigmoid(pos_cf_score - neg_cf_score))
        cf_loss = tf.negative(tf.reduce_mean(maxi))
        cf_reg_loss = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)

        return cf_loss, cf_reg_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

