# coding: utf-8
r"""
KGAT — Knowledge Graph Attention Network (structural only, no multimodal)
#########################################################################
Paper:
    Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation"
    KDD 2019 — https://arxiv.org/abs/1905.07854

Faithful (lunablack-style) MMRec port, without multimodal features:
  - Joint user+entity embedding table (CKG).
  - Decreasing per-layer dims via conv_dim_list (64 -> 32 -> 16).
  - bi-interaction aggregation through sparse A_in (torch.sparse.mm).
  - Layer-0 ego kept RAW, deeper layers L2-normalised, all concatenated.
  - A_in initialised from the summed normalised Laplacians, then recomputed
    once per epoch via TransR attention (update_attention).

Designed to run with MKGATTrainer (common/trainer.py), which calls:
    model(interaction, mode='train_cf')            → calc_cf_loss
    model.calc_kg_loss(h, r, pos_t, neg_t)         → KG TransR loss
    model.update_attention(h, t, r, relations)     → recompute A_in
    model.full_sort_predict(interaction)           → score matrix
"""

import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss
from utils.triplet import Triplet, Triplets


# ─────────────────────────────────────────────────────────────────────────────
# Graph propagation layer (bi-interaction)
# ─────────────────────────────────────────────────────────────────────────────

class Aggregator(nn.Module):
    """
    One KGAT bi-interaction propagation layer (paper §3.2):

        side = A_in @ ego
        out  = LeakyReLU(W1(ego + side)) + LeakyReLU(W2(ego * side))

    Aggregation is done with a sparse matrix product (torch.sparse.mm).
    Dropout is applied to the layer output.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

    def forward(self, ego_emb: torch.Tensor, A_in: torch.Tensor) -> torch.Tensor:
        side_emb = torch.sparse.mm(A_in, ego_emb)
        sum_part = self.activation(self.linear1(ego_emb + side_emb))
        bi_part  = self.activation(self.linear2(ego_emb * side_emb))
        out = sum_part + bi_part
        return self.message_dropout(out)


# ─────────────────────────────────────────────────────────────────────────────
# KGAT model
# ─────────────────────────────────────────────────────────────────────────────

class KGAT(GeneralRecommender):
    r"""Knowledge Graph Attention Network — structural variant (no multimodal)."""

    def __init__(self, config, dataset, triplets):
        super().__init__(config, dataset)

        # ── ìHyperparameters ──────────────────────────────────────────────────
        self.embed_dim      = config['embedding_size']
        self.relation_dim   = config['relation_dim']
        self.mess_dropout    = config['mess_dropout']            
        self.reg_weight      = config['reg_weight']
        self.kg_loss_weight  = (config['kg_loss_weight']
                                if config['kg_loss_weight'] is not None else 1.0)

        self.triplets = triplets

        # Per-layer dims (decreasing). '[32, 16]' with embed 64 → 64,32,16.
        if config['conv_dim_list'] is not None:
            self.conv_dim_list = [self.embed_dim] + eval(config['conv_dim_list'])
        else:
            self.conv_dim_list = [self.embed_dim] * (config['n_layers'] + 1)
        self.n_layers = len(self.conv_dim_list) - 1

        self._init_model()
        self.A_in.data = self._create_adjacency_matrix()
        self.mf_loss = BPRLoss()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_model(self):
        """Allocate all parameters and the propagation index structures."""
        all_node_ids = (
            [int(t.head) for t in self.triplets.data]
            + [int(t.tail) for t in self.triplets.data]
        )
        self.total_n_nodes = max(max(all_node_ids) + 1, self.n_users + self.n_items)

        all_rels = self.triplets.get_unique_relations()
        self.n_relations = len(all_rels)

        self.entity_user_embed = nn.Embedding(self.total_n_nodes, self.embed_dim)
        self.relation_embed    = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(
            torch.empty(self.n_relations, self.embed_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.A_in = nn.Parameter(
            torch.sparse_coo_tensor(size=(self.total_n_nodes, self.total_n_nodes)))
        self.A_in.requires_grad = False

        self.relation_dict = defaultdict(list)
        for tr in self.triplets.data:
            h, r, t = int(tr.head), int(tr.relation), int(tr.tail)
            self.relation_dict[r].append((h, t))

        item_node_ids = torch.arange(self.n_items, dtype=torch.long) + self.n_users
        self.register_buffer('item_node_ids', item_node_ids)

        # Aggregator layers with decreasing dims.
        if isinstance(self.mess_dropout, (list, tuple)):
            drop = list(self.mess_dropout)
            while len(drop) < self.n_layers:
                drop.append(drop[-1])
        else:
            drop = [float(self.mess_dropout)] * self.n_layers
        self.aggregator_layers = nn.ModuleList([
            Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], drop[k])
            for k in range(self.n_layers)
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # A_in construction (sum of per-relation normalised Laplacians)
    # ─────────────────────────────────────────────────────────────────────────

    def _norm_lap(self, adj):
        """Symmetric normalisation: D^-1/2 A D^-1/2."""
        rowsum = np.array(adj.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        return d_mat.dot(adj).dot(d_mat).tocoo()

    def _convert_coo2tensor(self, coo):
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values  = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(
            indices, values, torch.Size(coo.shape)).coalesce()

    def _create_adjacency_matrix(self):
        """
        Build the initial A_in as the sum of per-relation, symmetrically
        normalised adjacency matrices. Triples already include inverse
        relations, so edges are used as-is.
        """
        n = self.total_n_nodes
        A_sum = None
        for r, ht_list in self.relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1.0] * len(rows)
            adj  = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
            lap  = self._norm_lap(adj)
            A_sum = lap if A_sum is None else (A_sum + lap)
        return self._convert_coo2tensor(A_sum.tocoo())

    # ─────────────────────────────────────────────────────────────────────────
    # CF propagation
    # ─────────────────────────────────────────────────────────────────────────

    def calc_cf_embeddings(self):
        """Propagate over A_in; concat layer-0 (raw) + normalised deeper layers."""
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]                              # layer 0: raw
        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))
        return torch.cat(all_embed, dim=1)                   # (n_nodes, sum dims)

    # ─────────────────────────────────────────────────────────────────────────
    # Losses
    # ─────────────────────────────────────────────────────────────────────────

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(x.pow(2), dim=1, keepdim=False) / 2.0)

    def calc_cf_loss(self, interaction):
        """
        interaction[0]: user ids        (local, == node id)
        interaction[1]: pos item ids    (local, → node id + n_users)
        interaction[2]: neg item ids    (local, → node id + n_users)
        """
        user_ids     = interaction[0]
        item_pos_ids = interaction[1] + self.n_users
        item_neg_ids = interaction[2] + self.n_users

        all_embed = self.calc_cf_embeddings()
        user_embed     = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_score - neg_score))
        l2_loss = (self._L2_loss_mean(user_embed)
                   + self._L2_loss_mean(item_pos_embed)
                   + self._L2_loss_mean(item_neg_embed))
        return cf_loss + self.reg_weight * l2_loss

    def compute_kg_loss(self, h, r, pos_t, neg_t):
        """TransR pairwise ranking loss on a batch of triples."""
        r_embed = self.relation_embed(r)                              
        W_r = self.trans_M[r]                                         

        h_embed     = self.entity_user_embed(h)                       
        pos_t_embed = self.entity_user_embed(pos_t)
        neg_t_embed = self.entity_user_embed(neg_t)

        r_mul_h     = torch.bmm(h_embed.unsqueeze(1),     W_r).squeeze(1)   
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum((r_mul_h + r_embed - r_mul_pos_t).pow(2), dim=1)
        neg_score = torch.sum((r_mul_h + r_embed - r_mul_neg_t).pow(2), dim=1)

        kg_loss = torch.mean((-1.0) * F.logsigmoid(neg_score - pos_score))
        l2_loss = (self._L2_loss_mean(r_mul_h)
                   + self._L2_loss_mean(r_embed)
                   + self._L2_loss_mean(r_mul_pos_t)
                   + self._L2_loss_mean(r_mul_neg_t))
        return kg_loss + self.reg_weight * l2_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Attention update (once per epoch)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]          
        W_r = self.trans_M[r_idx]                           
        h_embed = self.entity_user_embed.weight[h_list]     
        t_embed = self.entity_user_embed.weight[t_list]
        r_mul_h = torch.matmul(h_embed, W_r)                 
        r_mul_t = torch.matmul(t_embed, W_r)
        return torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)

    @torch.no_grad()
    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device
        rows, cols, values = [], [], []
        for r_idx in relations:
            idx = torch.where(r_list == r_idx)
            batch_h = h_list[idx]
            batch_t = t_list[idx]
            batch_v = self._update_attention_batch(batch_h, batch_t, r_idx)
            rows.append(batch_h)
            cols.append(batch_t)
            values.append(batch_v)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        A_in = torch.sparse_coo_tensor(indices, values, torch.Size(self.A_in.shape))
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def full_sort_predict(self, interaction):
        """Return (batch_users, n_items) score matrix."""
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[interaction[0]]
        item_embed = all_embed[self.item_node_ids]
        return torch.matmul(user_embed, item_embed.transpose(0, 1))

    def calc_score(self, interaction):
        return self.full_sort_predict(interaction)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward dispatch (used by MKGATTrainer)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.compute_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)
        raise ValueError(f"Unknown forward mode: '{mode}'")
