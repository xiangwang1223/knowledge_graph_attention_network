import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class CKE(nn.Module):

    def __init__(self, args,
                 n_users, n_items, n_entities, n_relations, n_vocab,
                 user_pre_embed=None, item_pre_embed=None):

        super(CKE, self).__init__()

        # basic
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.n_vocab = n_vocab
        self.image_height = args.image_height
        self.image_width = args.image_width

        self.use_pretrain = args.use_pretrain

        # Structural Embedding (kg)
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        # Textual Embedding
        self.sdae_loss_fn = nn.MSELoss()

        self.sdae_encoder = nn.Sequential()
        sdae_encoder_dim_list = [n_vocab] + args.sdae_dim_list + [self.embed_dim]
        for idx in range(len(sdae_encoder_dim_list) - 1):
            in_dim = sdae_encoder_dim_list[idx]
            out_dim = sdae_encoder_dim_list[idx + 1]
            self.sdae_encoder.add_module('sdae_encoder_{idx}'.format(idx=idx), nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid()))

        self.sdae_decoder = nn.Sequential()
        sdae_decoder_dim_list = [self.embed_dim] + args.sdae_dim_list[::-1] + [n_vocab]
        for idx in range(len(sdae_decoder_dim_list) - 1):
            in_dim = sdae_decoder_dim_list[idx]
            out_dim = sdae_decoder_dim_list[idx + 1]
            self.sdae_decoder.add_module('sdae_decoder_{idx}'.format(idx=idx), nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid()))

        # Visual Embedding
        self.scae_loss_fn = nn.MSELoss()

        self.scae_encoder = nn.Sequential()
        scae_encoder_channel_list = [3] + args.scae_channel_list
        scae_encoder_kernel_list = args.scae_kernel_list
        for idx in range(len(scae_encoder_channel_list) - 1):
            in_channels = scae_encoder_channel_list[idx]
            out_channels = scae_encoder_channel_list[idx + 1]
            kernel_size = scae_encoder_kernel_list[idx]
            padding = int((kernel_size - 1) / 2)
            self.scae_encoder.add_module('scae_encoder_{idx}'.format(idx=idx), nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.Sigmoid()))

        self.scae_full_connect_encoder = nn.Sequential(nn.Linear(scae_encoder_channel_list[-1] * self.image_height * self.image_width, self.embed_dim), nn.Sigmoid())
        self.scae_full_connect_decoder = nn.Sequential(nn.Linear(self.embed_dim, scae_encoder_channel_list[-1] * self.image_height * self.image_width), nn.Sigmoid())

        self.scae_decoder = nn.Sequential()
        scae_decoder_channel_list = args.scae_channel_list[::-1] + [3]
        scae_decoder_kernel_list = args.scae_kernel_list[::-1]
        for idx in range(len(scae_decoder_channel_list) - 1):
            in_channels = scae_decoder_channel_list[idx]
            out_channels = scae_decoder_channel_list[idx + 1]
            kernel_size = scae_decoder_kernel_list[idx]
            padding = int((kernel_size - 1) / 2)
            self.scae_decoder.add_module('scae_decoder_{idx}'.format(idx=idx), nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.Sigmoid()))

        # cf
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embed_dim)

        if (self.use_pretrain == 1) and (user_pre_embed is not None):
            self.user_embed.weight = nn.Parameter(user_pre_embed)
        else:
            nn.init.xavier_uniform_(self.user_embed.weight)
        if (self.use_pretrain == 1) and (item_pre_embed is not None):
            self.item_embed.weight = nn.Parameter(item_pre_embed)
        else:
            nn.init.xavier_uniform_(self.item_embed.weight)


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        Structural Embedding
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                 # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                            # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_embed(h)                   # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_embed(pos_t)           # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_embed(neg_t)           # (kg_batch_size, embed_dim)

        # Equation (2)
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        r_embed = F.normalize(r_embed, p=2, dim=1)
        r_mul_h = F.normalize(r_mul_h, p=2, dim=1)
        r_mul_pos_t = F.normalize(r_mul_pos_t, p=2, dim=1)
        r_mul_neg_t = F.normalize(r_mul_neg_t, p=2, dim=1)

        # Equation (3)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def calc_sdae_loss(self, masked_textual_embed, textual_embed):
        """
        Textual Embedding
        masked_textual_embed:    (sdae_batch_size, n_vocab)
        textual_embed:           (sdae_batch_size, n_vocab)
        """
        encoded_embed = self.sdae_encoder(masked_textual_embed)
        decoded_embed = self.sdae_decoder(encoded_embed)
        sdae_loss = self.sdae_loss_fn(decoded_embed, textual_embed)
        return sdae_loss


    def calc_scae_loss(self, masked_visual_embed, visual_embed):
        """
        Visual Embedding
        masked_visual_embed:    (scae_batch_size, raw_channel=3, image_height, image_width)
        visual_embed:           (scae_batch_size, raw_channel=3, image_height, image_width)
        """
        encoded_embed = self.scae_encoder(masked_visual_embed)              # (scae_batch_size, encoder_channel, image_height, image_width)
        scae_batch_size, encoder_channel, image_height, image_width = encoded_embed.shape
        encoded_embed = encoded_embed.view([scae_batch_size, -1])
        encoded_embed = self.scae_full_connect_encoder(encoded_embed)       # (scae_batch_size, embed_dim)

        decoded_embed = self.scae_full_connect_decoder(encoded_embed)
        decoded_embed = decoded_embed.view([scae_batch_size, encoder_channel, image_height, image_width])
        decoded_embed = self.scae_decoder(decoded_embed)

        scae_loss = self.scae_loss_fn(decoded_embed, visual_embed)
        return scae_loss


    def generate_item_cf_embed(self, item_ids, item_textual_embed, item_visual_embed):
        """
        item_ids:           (cf_batch_size)
        item_textual_embed: (cf_batch_size, n_vocab)
        item_visual_embed:  (cf_batch_size, raw_channel=3, image_height, image_width)
        """
        item_embed = self.item_embed(item_ids)                              # (cf_batch_size, embed_dim)

        item_kg_embed = self.entity_embed(item_ids)                         # (cf_batch_size, embed_dim)

        item_sdae_embed = self.sdae_encoder(item_textual_embed)             # (cf_batch_size, embed_dim)

        item_scae_embed = self.scae_encoder(item_visual_embed)
        scae_batch_size, _, _, _ = item_scae_embed.shape
        item_scae_embed = item_scae_embed.view([scae_batch_size, -1])
        item_scae_embed = self.scae_full_connect_encoder(item_scae_embed)   # (cf_batch_size, embed_dim)

        # Equation (5)
        item_cf_embed = item_embed + item_kg_embed + item_sdae_embed + item_scae_embed
        return item_cf_embed


    def calc_cf_loss(self,
                     user_ids,
                     item_pos_ids, item_neg_ids,
                     item_pos_textual_embed, item_neg_textual_embed,
                     item_pos_visual_embed, item_neg_visual_embed):
        """
        user_ids:       (cf_batch_size)

        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)

        item_pos_textual_embed:  (cf_batch_size, n_vocab)
        item_neg_textual_embed:  (cf_batch_size, n_vocab)

        item_pos_visual_embed:   (cf_batch_size, raw_channel=3, image_height, image_width)
        item_neg_visual_embed:   (cf_batch_size, raw_channel=3, image_height, image_width)
        """
        user_embed = self.user_embed(user_ids)                                                                          # (cf_batch_size, embed_dim)
        item_pos_cf_embed = self.generate_item_cf_embed(item_pos_ids, item_pos_textual_embed, item_pos_visual_embed)    # (cf_batch_size, embed_dim)
        item_neg_cf_embed = self.generate_item_cf_embed(item_neg_ids, item_neg_textual_embed, item_neg_visual_embed)    # (cf_batch_size, embed_dim)

        # Equation (6)
        pos_score = torch.sum(user_embed * item_pos_cf_embed, dim=1)    # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_cf_embed, dim=1)    # (cf_batch_size)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_cf_embed) + _L2_loss_mean(item_neg_cf_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def calc_loss(self,
                  h, r, pos_t, neg_t,
                  masked_textual_embed, textual_embed,
                  masked_visual_embed, visual_embed,
                  user_ids, item_pos_ids, item_neg_ids, item_pos_textual_embed, item_neg_textual_embed, item_pos_visual_embed, item_neg_visual_embed):
        """
        h:          (kg_batch_size)
        r:          (kg_batch_size)
        pos_t:      (kg_batch_size)
        neg_t:      (kg_batch_size)

        masked_textual_embed:    (sdae_batch_size, n_vocab)
        textual_embed:           (sdae_batch_size, n_vocab)

        masked_visual_embed:     (scae_batch_size, raw_channel=3, image_height, image_width)
        visual_embed:            (scae_batch_size, raw_channel=3, image_height, image_width)

        user_ids:                (cf_batch_size)
        item_pos_ids:            (cf_batch_size)
        item_neg_ids:            (cf_batch_size)
        item_pos_textual_embed:  (cf_batch_size, n_vocab)
        item_neg_textual_embed:  (cf_batch_size, n_vocab)
        item_pos_visual_embed:   (cf_batch_size, raw_channel=3, image_height, image_width)
        item_neg_visual_embed:   (cf_batch_size, raw_channel=3, image_height, image_width)
        """
        kg_loss = self.calc_kg_loss(h, r, pos_t, neg_t)
        sdae_loss = self.calc_sdae_loss(masked_textual_embed, textual_embed)
        scae_loss = self.calc_scae_loss(masked_visual_embed, visual_embed)
        cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids, item_pos_textual_embed, item_neg_textual_embed, item_pos_visual_embed, item_neg_visual_embed)
        loss = kg_loss + sdae_loss + scae_loss + cf_loss
        return loss


    def calc_score(self, user_ids, item_ids, item_textual_embed, item_visual_embed):
        """
        user_ids:           (n_users)
        item_ids:           (n_items)
        item_textual_embed: (n_items, n_vocab)
        item_visual_embed:  (n_items, raw_channel=3, image_height, image_width)
        """
        user_embed = self.user_embed(user_ids)                                                          # (n_users, embed_dim)
        item_cf_embed = self.generate_item_cf_embed(item_ids, item_textual_embed, item_visual_embed)    # (n_items, embed_dim)
        cf_score = torch.matmul(user_embed, item_cf_embed.transpose(0, 1))                              # (n_users, n_items)
        return cf_score


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)



if __name__ == '__main__':
    import argparse
    import numpy as np
    import torch.optim as optim

    args = argparse.ArgumentParser().parse_args()

    # basic
    n_users = 81
    n_items = 103
    n_entities = 189
    n_relations = 7
    n_vocab = 880
    args.use_pretrain = 0

    # Structural Embedding
    args.embed_dim = 64
    args.relation_dim = 50
    args.kg_l2loss_lambda = 1e-5

    # Textual Embedding
    args.sdae_dim_list = [300, 100]

    # Visual Embedding
    args.image_height = 270
    args.image_width = 280

    args.scae_channel_list = [6, 9]
    args.scae_kernel_list = [5, 3]

    # cf
    args.cf_l2loss_lambda = 1e-5

    # Structural Embedding training input
    kg_batch_size = 11

    h = torch.LongTensor(np.random.randint(n_entities, size=kg_batch_size))
    r = torch.LongTensor(np.random.randint(n_relations, size=kg_batch_size))
    pos_t = torch.LongTensor(np.random.randint(n_entities, size=kg_batch_size))
    neg_t = torch.LongTensor(np.random.randint(n_entities, size=kg_batch_size))

    # Textual Embedding training input
    vocab_occu_max = 5
    sdae_masked_dropout = 0.2

    sdae_batch_size = 22
    textual_embed = torch.FloatTensor(np.random.randint(vocab_occu_max, size=[sdae_batch_size, n_vocab])) / vocab_occu_max
    textual_mask_bool = torch.BoolTensor(np.random.random([sdae_batch_size, n_vocab]) >= sdae_masked_dropout)
    masked_textual_embed = textual_embed * textual_mask_bool

    # Visual Embedding training input
    image_channel = 3
    rbg_value_max = 256
    scae_masked_dropout = 0.2

    scae_batch_size = 33
    visual_embed = torch.FloatTensor(np.random.randint(rbg_value_max, size=[scae_batch_size, image_channel, args.image_height, args.image_width])) / rbg_value_max
    visual_mask_bool = torch.BoolTensor(np.random.random([scae_batch_size, image_channel, args.image_height, args.image_width]) >= scae_masked_dropout)
    masked_visual_embed = visual_embed * visual_mask_bool

    # cf training input
    cf_batch_size = 44

    user_ids = torch.LongTensor(np.random.randint(n_users, size=cf_batch_size))
    item_pos_ids = torch.LongTensor(np.random.randint(n_items, size=cf_batch_size))
    item_neg_ids = torch.LongTensor(np.random.randint(n_items, size=cf_batch_size))

    item_pos_textual_embed = torch.FloatTensor(np.random.randint(vocab_occu_max, size=[cf_batch_size, n_vocab])) / vocab_occu_max
    item_neg_textual_embed = torch.FloatTensor(np.random.randint(vocab_occu_max, size=[cf_batch_size, n_vocab])) / vocab_occu_max

    item_pos_visual_embed = torch.FloatTensor(np.random.randint(rbg_value_max, size=[cf_batch_size, image_channel, args.image_height, args.image_width])) / rbg_value_max
    item_neg_visual_embed = torch.FloatTensor(np.random.randint(rbg_value_max, size=[cf_batch_size, image_channel, args.image_height, args.image_width])) / rbg_value_max

    # build mode
    model = CKE(args, n_users, n_items, n_entities, n_relations, n_vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    # calculate loss
    loss = model(h, r, pos_t, neg_t,
                 masked_textual_embed, textual_embed,
                 masked_visual_embed, visual_embed,
                 user_ids, item_pos_ids, item_neg_ids, item_pos_textual_embed, item_neg_textual_embed, item_pos_visual_embed, item_neg_visual_embed,
                 is_train=True)
    print(loss.item())

    # train model
    for iter in range(30):
        loss = model(h, r, pos_t, neg_t,
                     masked_textual_embed, textual_embed,
                     masked_visual_embed, visual_embed,
                     user_ids, item_pos_ids, item_neg_ids, item_pos_textual_embed, item_neg_textual_embed, item_pos_visual_embed, item_neg_visual_embed,
                     is_train=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Iter {}: '.format(iter + 1), loss.item())

    # cf predicting input
    user_ids = torch.LongTensor(np.arange(n_users))
    item_ids = torch.LongTensor(np.arange(n_items))
    item_textual_embed = torch.FloatTensor(np.random.randint(vocab_occu_max, size=[n_items, n_vocab])) / vocab_occu_max
    item_visual_embed = torch.FloatTensor(np.random.randint(rbg_value_max, size=[n_items, image_channel, args.image_height, args.image_width])) / rbg_value_max

    # predict
    with torch.no_grad():
        cf_score = model(user_ids, item_ids, item_textual_embed, item_visual_embed, is_train=False)
    print(cf_score.numpy())


