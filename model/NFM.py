import torch
from torch import nn
from torch.nn import functional as F


class HiddenLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):
        super(HiddenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.message_dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.message_dropout(out)
        return out


class NFM(nn.Module):

    def __init__(self, args,
                 n_users, n_items, n_entities,
                 user_pre_embed=None, item_pre_embed=None):

        super(NFM, self).__init__()
        self.model_type = args.model_type
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_features = n_users + n_entities

        self.embed_dim = args.embed_dim
        self.l2loss_lambda = args.l2loss_lambda

        self.hidden_dim_list = [args.embed_dim] + eval(args.hidden_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.hidden_dim_list))

        self.linear = nn.Linear(self.n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            user_entity_embed = torch.cat([user_pre_embed, item_pre_embed, other_entity_embed], dim=0)
            self.feature_embed = nn.Parameter(user_entity_embed)
        else:
            self.feature_embed = nn.Parameter(torch.Tensor(self.n_features, self.embed_dim))
            nn.init.xavier_uniform_(self.feature_embed)

        if self.model_type == 'fm':
            self.h = nn.Linear(self.embed_dim, 1, bias=False)
            with torch.no_grad():
                self.h.weight.copy_(torch.ones([1, self.embed_dim]))
            for param in self.h.parameters():
                param.requires_grad = False

        elif self.model_type == 'nfm':
            self.hidden_layers = nn.ModuleList()
            for idx in range(self.n_layers):
                self.hidden_layers.append(HiddenLayer(self.hidden_dim_list[idx], self.hidden_dim_list[idx + 1], self.mess_dropout[idx]))
            self.h = nn.Linear(self.hidden_dim_list[-1], 1, bias=False)
            nn.init.xavier_uniform_(self.h.weight)


    def calc_score(self, feature_values):
        """
        feature_values:  (batch_size, n_features), n_features = n_users + n_entities, torch.sparse.FloatTensor
        """
        # Bi-Interaction layer
        # Equation (4) / (3)
        sum_square_embed = torch.mm(feature_values, self.feature_embed).pow(2)           # (batch_size, embed_dim)
        square_sum_embed = torch.mm(feature_values.pow(2), self.feature_embed.pow(2))    # (batch_size, embed_dim)
        z = 0.5 * (sum_square_embed - square_sum_embed)                                  # (batch_size, embed_dim)

        if self.model_type == 'nfm':
            # Equation (5)
            for i, layer in enumerate(self.hidden_layers):
                z = layer(z)                                # (batch_size, hidden_dim)

        # Prediction layer
        # Equation (6)
        y = self.h(z)                                       # (batch_size, 1)
        # Equation (2) / (7) / (8)
        y = self.linear(feature_values) + y                 # (batch_size, 1)
        return y.squeeze()                                  # (batch_size)


    def calc_loss(self, pos_feature_values, neg_feature_values):
        """
        pos_feature_values:  (batch_size, n_features), torch.sparse.FloatTensor
        neg_feature_values:  (batch_size, n_features), torch.sparse.FloatTensor
        """
        pos_scores = self.calc_score(pos_feature_values)            # (batch_size)
        neg_scores = self.calc_score(neg_feature_values)            # (batch_size)

        loss = (-1.0) * torch.log(1e-10 + F.sigmoid(pos_scores - neg_scores))
        loss = torch.mean(loss)

        l2_loss = torch.norm(self.h.weight, 2).pow(2) / 2
        loss += self.l2loss_lambda * l2_loss
        return loss


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)


