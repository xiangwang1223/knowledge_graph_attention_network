'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, CFKG, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class CFKG(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        self._build_inputs()
        self.weights = self._build_weights()

        self._build_model()
        self._build_loss()

        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = 'cfkg'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100

        self.margin = 1.0
        self.L1_flag = args.l1_flag

        self.lr = args.lr
        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose

    def _build_inputs(self):
        # placeholder definition

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        # dropout: node dropout (adopted on the ego-networks); message dropout (adopted on the convolution operations).
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer(uniform = False)

        if self.pretrain_data is None:
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]), name='entity_embed')
            print('using xavier initialization')
        else:
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)

            item_embed = self.pretrain_data['item_embed']
            other_embed = initializer([self.n_entities - self.n_items, self.emb_dim])

            all_weights['entity_embed'] = tf.Variable(initial_value=tf.concat([item_embed, other_embed], 0),
                                                      trainable=True, name='entity_embed', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.emb_dim]),
                                                    name='relation_embed')

        return all_weights


    def _build_model(self):
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)


        # self.batch_predictions = tf.matmul(self.h_e + self.r_e, self.pos_t_e, transpose_a=False, transpose_b=True)
        # self.batch_predictions = tf.reduce_sum((self.h_e + self.r_e - self.pos_t_e) ** 2, 1, keepdims = True)

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)

        # head & tail entity embeddings: batch_size *1 * emb_dim
        h_e = tf.nn.embedding_lookup(embeddings, h)
        pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        # neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _build_loss(self):
        if self.L1_flag:
            pos_kg_score = tf.reduce_sum(abs(self.h_e + self.r_e - self.pos_t_e), 1, keepdims = True)
            neg_kg_score = tf.reduce_sum(abs(self.h_e + self.r_e - self.neg_t_e), 1, keepdims = True)

            self.batch_predictions = - pos_kg_score
        else:
            pos_kg_score = tf.reduce_sum((self.h_e + self.r_e - self.pos_t_e) ** 2, 1, keepdims=True)
            neg_kg_score = tf.reduce_sum((self.h_e + self.r_e - self.neg_t_e) ** 2, 1, keepdims=True)

            self.batch_predictions = - pos_kg_score

        kg_loss = tf.reduce_mean(tf.maximum(pos_kg_score - neg_kg_score + self.margin, 0))

        # kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
        #               tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        # kg_reg_loss = kg_reg_loss / self.batch_size

        self.reg_loss = tf.constant(0.0, tf.float32, [1])
        self.base_loss = tf.constant(0.0, tf.float32, [1])
        self.kge_loss = kg_loss
        # self.reg_loss = self.regs[0] * kg_reg_loss
        self.loss = self.kge_loss + self.reg_loss

        # Optimization process.
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

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