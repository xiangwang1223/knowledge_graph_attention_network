import torch
import torch.nn as nn
import torch.nn.functional as F


class ECFKG(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(ECFKG, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim

        self.relation_embed = nn.Embedding(self.n_relations, self.embed_dim)
        nn.init.xavier_normal_(self.relation_embed.weight)

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_normal_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_normal_(self.entity_user_embed.weight)


    def calc_score(self, user_ids, item_ids, relation_id):
        """
        user_ids:     (n_users)
        item_ids:     (n_items)
        relation_id:  (1)
        """
        r_embed = self.relation_embed(relation_id)          # (1, embed_dim)
        user_embed = self.entity_user_embed(user_ids)       # (n_users, embed_dim)
        item_embed = self.entity_user_embed(item_ids)       # (n_items, embed_dim)

        cf_score = torch.matmul((user_embed + r_embed), item_embed.transpose(0, 1))     # (n_users, n_items)
        # cf_score = F.softmax(cf_score, dim=1)
        return cf_score


    def calc_loss(self, h, r, pos_t, neg_t):
        """
        h:      (batch_size)
        r:      (batch_size)
        pos_t:  (batch_size)
        neg_t:  (batch_size)
        """
        r_embed = self.relation_embed(r)                 # (batch_size, embed_dim)
        h_embed = self.entity_user_embed(h)              # (batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)      # (batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)      # (batch_size, embed_dim)

        pos_score = torch.sum((h_embed + r_embed) * pos_t_embed, dim=1)      # (batch_size)
        neg_score = torch.sum((h_embed + r_embed) * neg_t_embed, dim=1)      # (batch_size)

        kg_loss = (F.softplus(-pos_score) + F.softplus(neg_score)).mean()
        return kg_loss


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)


