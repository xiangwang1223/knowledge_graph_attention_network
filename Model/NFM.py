'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline model, NFM, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class NFM(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model()
        self._build_loss()
        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        if args.model_type == 'nfm':
            self.model_type = 'nfm'
        else:
            self.model_type = 'fm'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']

        self.n_features = data_config['n_users'] + data_config['n_entities']

        self.lr = args.lr
        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for neural CF part.
        if args.model_type == 'nfm':
            self.weight_size = eval(args.layer_size)
            self.n_layers = len(self.weight_size)

            self.model_type += '_l%d' % self.n_layers
        else:
            self.weight_size = []
            self.n_layers = 0

        self.regs = eval(args.regs)

        self.verbose = args.verbose

    def _build_inputs(self):
        self.pos_indices = tf.placeholder(tf.int64, shape=[None, 2], name='pos_indices')
        self.pos_values = tf.placeholder(tf.float32, shape=[None], name='pos_values')
        self.pos_shape = tf.placeholder(tf.int64, shape=[2], name='pos_shape')

        self.neg_indices = tf.placeholder(tf.int64, shape=[None, 2], name='neg_indices')
        self.neg_values = tf.placeholder(tf.float32, shape=[None], name='neg_values')
        self.neg_shape = tf.placeholder(tf.int64, shape=[2], name='neg_shape')

        self.mess_dropout = tf.placeholder(tf.float32, shape=[None], name='mess_dropout')

        # Input positive features, shape=(batch_size * feature_dim)
        self.sp_pos_feats = tf.SparseTensor(self.pos_indices, self.pos_values, self.pos_shape)
        # Input negative features, shape=(batch_size * feature_dim)
        self.sp_neg_feats = tf.SparseTensor(self.neg_indices, self.neg_values, self.neg_shape)

    def _build_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['var_linear'] = tf.Variable(initializer([self.n_features, 1]), name='var_linear')

        # model parameters for FM.
        if self.pretrain_data is None:
            all_weights['var_factor'] = tf.Variable(initializer([self.n_features, self.emb_dim]), name='var_factor')
            print('using xavier initialization')
        else:
            user_embed = self.pretrain_data['user_embed']
            item_embed = self.pretrain_data['item_embed']
            other_embed = initializer([self.n_entities - self.n_items, self.emb_dim])

            all_weights['var_factor'] = tf.Variable(initial_value=tf.concat([user_embed, item_embed, other_embed], 0),
                                                    trainable=True, name='var_factor', dtype=tf.float32)

            # user_embed = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True, dtype=tf.float32)
            # item_embed = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True, dtype=tf.float32)
            # other_embed = tf.Variable(initializer([self.n_entities - self.n_items, self.emb_dim]))
            #
            # all_weights['var_factor'] = tf.concat([user_embed, item_embed, other_embed], 0, name='var_factor')
            print('using pretrained initialization')

        # model parameters for NFM.
        self.weight_size_list = [self.emb_dim] + self.weight_size
        for i in range(self.n_layers):
            all_weights['W_%d' %i] = tf.Variable(
                initializer([self.weight_size_list[i], self.weight_size_list[i+1]]), name='W_%d' %i)
            all_weights['b_%d' %i] = tf.Variable(
                initializer([1, self.weight_size_list[i+1]]), name='b_%d' %i)

        if self.model_type == 'fm':
            all_weights['h'] = tf.constant(1., tf.float32, [self.emb_dim, 1])
        else:
            all_weights['h'] = tf.Variable(initializer([self.weight_size_list[-1], 1]), name='h')

        return all_weights

    def _build_model(self):
        self.batch_predictions = self._get_bi_pooling_predictions(self.sp_pos_feats)

    def _build_loss(self):
        pos_scores = self._get_bi_pooling_predictions(self.sp_pos_feats)
        neg_scores = self._get_bi_pooling_predictions(self.sp_neg_feats)

        maxi = tf.log(1e-10 + tf.nn.sigmoid(pos_scores - neg_scores))
        cf_loss = tf.negative(tf.reduce_mean(maxi))

        self.base_loss = cf_loss
        self.reg_loss = self.regs[0] * tf.nn.l2_loss(self.weights['h'])
        # self.reg_loss = self.regs[0] * tf.nn.l2_loss(self.weights['var_factor']) + \
        #                 self.regs[1] * tf.nn.l2_loss(self.weights['h'])
        #
        # for k in range(self.n_layers):
        #     self.reg_loss += self.regs[-1] * (tf.nn.l2_loss(self.weights['W_%d' % k]))

        self.kge_loss = tf.constant(0.0, tf.float32, [1])

        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _get_bi_pooling_predictions(self, feats):
        # Linear terms: batch_size * 1
        term0 = tf.sparse_tensor_dense_matmul(feats, self.weights['var_linear'])

        # Interaction terms w.r.t. first sum then square: batch_size * emb_size.
        #   e.g., sum_{k from 1 to K}{(v1k+v2k)**2}
        sum_emb = tf.sparse_tensor_dense_matmul(feats, self.weights['var_factor'])
        term1 = tf.square(sum_emb)

        # Interaction terms w.r.t. first square then sum: batch_size * emb_size.
        #   e.g., sum_{k from 1 to K}{v1k**2 + v2k**2}
        square_emb = tf.sparse_tensor_dense_matmul(tf.square(feats), tf.square(self.weights['var_factor']))
        term2 = square_emb

        # "neural factorization machine", Equation 3, the result of bi-interaction pooling: batch_size * emb_size
        term3 = 0.5 * (term1 - term2)

        # "neural factorization machine", Equation 7, the result of MLP: batch_size * 1
        z = [term3]
        for i in range(self.n_layers):
            temp = tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i])
            temp = tf.nn.dropout(temp, 1 - self.mess_dropout[i])
            z.append(temp)

        preds = term0 + tf.matmul(z[-1], self.weights['h'])

        return preds

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