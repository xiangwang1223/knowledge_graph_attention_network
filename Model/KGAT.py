'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class KGAT(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self._build_inputs()

        """
        *********************************************************
        Create Model Parameters for CF & KGE parts.
        """
        self.weights = self._build_weights()

        """
        *********************************************************
        Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
        """
        self._build_model_phase_I()
        """
        Optimize Recommendation (CF) Part via BPR Loss.
        """
        self._build_loss_phase_I()

        """
        *********************************************************
        Compute Knowledge Graph Embeddings via TransR.
        """
        self._build_model_phase_II()
        """
        Optimize KGE Part via BPR Loss.
        """
        self._build_loss_phase_II()

        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'kgat'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100

        # initialize the attentive matrix A for phase I.
        self.A_in = data_config['A_in']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose

    def _build_inputs(self):
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # for knowledge graph modeling (TransD)
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        # dropout: node dropout (adopted on the ego-networks);
        # message dropout (adopted on the convolution operations).
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

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

        all_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                    name='relation_embed')
        all_weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]))

        self.weight_size_list = [self.emb_dim] + self.weight_size


        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _build_model_phase_I(self):
        if self.alg_type in ['bi']:
            self.ua_embeddings, self.ea_embeddings = self._create_bi_interaction_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ea_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['graphsage']:
            self.ua_embeddings, self.ea_embeddings = self._create_graphsage_embed()

        self.u_e = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.pos_items)
        self.neg_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.neg_items)

        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)

    def _build_model_phase_II(self):
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)
        self.A_kg_score = self._generate_transE_score(h=self.h, t=self.pos_t, r=self.r)
        self.A_out = self._create_attentive_A_out()

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        # head & tail entity embeddings: batch_size *1 * emb_dim
        h_e = tf.nn.embedding_lookup(embeddings, h)
        pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.kge_dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.kge_dim])

        h_e = tf.math.l2_normalize(h_e, axis=1)
        r_e = tf.math.l2_normalize(r_e, axis=1)
        pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _build_loss_phase_I(self):
        pos_scores = tf.reduce_sum(tf.multiply(self.u_e, self.pos_i_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_e, self.neg_i_e), axis=1)

        regularizer = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)
        regularizer = regularizer / self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        base_loss = tf.negative(tf.reduce_mean(maxi))

        self.base_loss = base_loss
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.regs[0] * regularizer
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.RMSPropOptimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_loss_phase_II(self):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)

        maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        kg_loss = tf.negative(tf.reduce_mean(maxi))
        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _create_bi_interaction_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_gcn_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_graphsage_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        pre_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            # line 1 in algorithm 1 [RM-GCN, KDD'2018], aggregator layer: weighted sum
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], pre_embeddings))
            embeddings = tf.concat(temp_embed, 0)

            # line 2 in algorithm 1 [RM-GCN, KDD'2018], aggregating the previsou embeddings
            embeddings = tf.concat([pre_embeddings, embeddings], 1)
            pre_embeddings = tf.nn.relu(
                tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k])

            pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A

    def _generate_transE_score(self, h, t, r):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.kge_dim])

        # l2-normalize
        h_e = tf.math.l2_normalize(h_e, axis=1)
        r_e = tf.math.l2_normalize(r_e, axis=1)
        t_e = tf.math.l2_normalize(t_e, axis=1)

        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)

        return kg_score

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

    def train_A(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

    """
    Update the attentive laplacian matrix.
    """
    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        kg_score = np.array(kg_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_entities,
                                                                       self.n_users + self.n_entities))
        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.)