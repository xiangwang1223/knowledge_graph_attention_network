import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class CKE(nn.Module):

    def __init__(self, args,
                 n_users, n_items, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(CKE, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embed_dim)

        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        if (self.use_pretrain == 1) and (user_pre_embed is not None):
            self.user_embed.weight = nn.Parameter(user_pre_embed)
        else:
            nn.init.xavier_uniform_(self.user_embed.weight)

        if (self.use_pretrain == 1) and (item_pre_embed is not None):
            self.item_embed.weight = nn.Parameter(item_pre_embed)
        else:
            nn.init.xavier_uniform_(self.item_embed.weight)

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
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


    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        user_embed = self.user_embed(user_ids)                          # (cf_batch_size, embed_dim)
        item_pos_embed = self.item_embed(item_pos_ids)                  # (cf_batch_size, embed_dim)
        item_neg_embed = self.item_embed(item_neg_ids)                  # (cf_batch_size, embed_dim)

        item_pos_kg_embed = self.entity_embed(item_pos_ids)             # (cf_batch_size, embed_dim)
        item_neg_kg_embed = self.entity_embed(item_neg_ids)             # (cf_batch_size, embed_dim)

        # Equation (5)
        item_pos_cf_embed = item_pos_embed + item_pos_kg_embed          # (cf_batch_size, embed_dim)
        item_neg_cf_embed = item_neg_embed + item_neg_kg_embed          # (cf_batch_size, embed_dim)

        # Equation (6)
        pos_score = torch.sum(user_embed * item_pos_cf_embed, dim=1)    # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_cf_embed, dim=1)    # (cf_batch_size)

        cf_loss = (-1.0) * torch.log(1e-10 + F.sigmoid(pos_score - neg_score))
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_cf_embed) + _L2_loss_mean(item_neg_cf_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids, h, r, pos_t, neg_t):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)

        h:              (kg_batch_size)
        r:              (kg_batch_size)
        pos_t:          (kg_batch_size)
        neg_t:          (kg_batch_size)
        """
        kg_loss = self.calc_kg_loss(h, r, pos_t, neg_t)
        cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
        loss = kg_loss + cf_loss
        return loss


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        user_embed = self.user_embed(user_ids)                  # (n_users, embed_dim)

        item_embed = self.item_embed(item_ids)                  # (n_items, embed_dim)
        item_kg_embed = self.entity_embed(item_ids)             # (n_items, embed_dim)
        item_cf_embed = item_embed + item_kg_embed              # (n_items, embed_dim)

        cf_score = torch.matmul(user_embed, item_cf_embed.transpose(0, 1))      # (n_users, n_items)
        return cf_score


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)


