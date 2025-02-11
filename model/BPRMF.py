import torch
from torch import nn
from torch.nn import functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class BPRMF(nn.Module):

    def __init__(self, args,
                 n_users, n_items,
                 user_pre_embed=None, item_pre_embed=None):

        super(BPRMF, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = args.embed_dim
        self.l2loss_lambda = args.l2loss_lambda

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


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:   (n_users)
        item_ids:   (n_items)
        """
        user_embed = self.user_embed(user_ids)                              # (n_users, embed_dim)
        item_embed = self.item_embed(item_ids)                              # (n_items, embed_dim)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))     # (n_users, n_items)
        return cf_score


    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (batch_size)
        item_pos_ids:   (batch_size)
        item_neg_ids:   (batch_size)
        """
        user_embed = self.user_embed(user_ids)              # (batch_size, embed_dim)
        item_pos_embed = self.item_embed(item_pos_ids)      # (batch_size, embed_dim)
        item_neg_embed = self.item_embed(item_neg_ids)      # (batch_size, embed_dim)

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)       # (batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)       # (batch_size)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.l2loss_lambda * l2_loss
        return loss


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)


