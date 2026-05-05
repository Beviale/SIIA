# coding: utf-8
r"""

################################################
paper:  Multi-modal Knowledge Graphs for Recommender Systems
https://dl.acm.org/doi/pdf/10.1145/3340531.3411947
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss
from utils.triplet import Triplet, Triplets


class MKGAT(GeneralRecommender):
    r"""MKGAT is a multi-modal knowledge graph based recommender model.
 
    MKGAT introduces multi-modal entities (images and text) as first-class citizens
    of the knowledge graph. It proposes a multi-modal graph attention technique to
    conduct information propagation over MMKGs, and uses the resulting aggregated
    embedding for recommendation.
 
    We implement the model following the original author with a pairwise training mode.
    """
 
    def __init__(self, config, dataset):
        super(MKGAT, self).__init__(config, dataset)

        # load configuration
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']  
        self.reg_weight = config['reg_weight'] 
        self.aggregation = config['aggregation_type']
        self.leaky_relu_slop = config['leaky_relu_slope']
        self.batch_size_triplets = config['batch_size_triplets']

        # load dataset info
        if config['model_enriched_triples_format'] and config['dataset_support_triplets']:
            self.triplets = dataset.kg_triplets()
        else:
            raise Exception("The model cannot be trained!")
        self._init_model()
 
    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────
 
    def _init_model(self):

        self.max_structural_entity_index = max(int(entity) for entity in self.triplets.get_unique_entities_and_users()) 
        self.max_structural_relation_index =  max(int(r) for r in self.triplets.get_unique_relations())

        self.n_structural_entities = len(self.triplets.get_unique_entities()) 
        self.entity_embedding = nn.Embedding(self.n_structural_entities, self.embedding_dim)
        

        self.n_relations = (self.max_structural_relation_index + 1) \
                 + (1 if self.v_feat is not None else 0) \
                 + (1 if self.t_feat is not None else 0)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_dim)
    
        self.n_triplets_users = len(self.triplets.get_unique_users()) 
        self.user_embedding = nn.Embedding(self.n_triplets_users, self.embedding_dim)


        self.shared_structural_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(self.leaky_relu_slop)
        )

        self.n_triplets_items= len(self.triplets.get_unique_items()) 
        new_triplets, n_valid_images, n_valid_texts = self._build_kg_triplets()
        if n_valid_images is not None and n_valid_images > 0:
           self.image_fc = nn.Sequential(
                nn.Linear(n_valid_images, self.embedding_dim),
                nn.LeakyReLU(self.leaky_relu_slop)
            )       
        if n_valid_texts is not None and n_valid_texts > 0:
           self.text_fc = nn.Sequential(
                nn.Linear(n_valid_texts, self.embedding_dim),
                nn.LeakyReLU(self.leaky_relu_slop)
            )
           
        self.triplets.extend(new_triplets)

        self.mf_loss = BPRLoss()

        # attention parameters (propagation layer)
        self.W1 = nn.Linear(self.embedding_dim * 3, self.embedding_dim)  # triple embedding
        self.W2 = nn.Linear(self.embedding_dim, 1)                    # attention score
 
        if self.aggregation == "add":
            self.W3 = nn.Linear(self.embedding_dim, self.embedding_dim)
        else:  
            self.W3 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)  

        self.valid_tails = {}
        for t in self.triplets.data:
            key = (t.head, t.relation)
            if key not in self.valid_tails:
                self.valid_tails[key] = set()
            self.valid_tails[key].add(t.tail)



    def get_entity_embedding(self, entity_id):
        entity_id_int = torch.tensor(int(entity_id), dtype=torch.long, device=self.device)
        if entity_id_int < self.n_triplets_users:
            return self.shared_structural_fc(self.user_embedding(torch.tensor(entity_id_int, device=self.device)))
        if entity_id_int < self.image_offset:
            return self.shared_structural_fc(self.entity_embedding(torch.tensor(entity_id_int - self.n_triplets_users, device=self.device)))
        elif entity_id_int < self.text_offset:
            idx = entity_id_int - self.image_offset
            return self.v_feat[idx]
        else:
            idx = entity_id_int - self.text_offset
            return self.t_feat[idx]
    

    def get_relation_embedding(self, relation_id):
        relation_id_int = torch.tensor(int(relation_id), dtype=torch.long, device=self.device)
        return self.shared_structural_fc(self.relation_embedding(torch.tensor(relation_id_int, device=self.device)))
    
    def _get_multimodal_head_embeddings(self):
        entities_emb = []
        head_to_idx = {}
        index = 0
        for entity in sorted(self.triplets.get_all_head_entities()):
            head_to_idx[entity] = index
            entities_emb.append(self.get_entity_embedding(entity))  
            index = index + 1
        self.head_to_idx = head_to_idx
        return torch.stack(entities_emb, dim=0)
    
    def _build_kg_triplets(self):
        item_ids = torch.tensor([int(x) for x in self.triplets.get_unique_items()], dtype=torch.long, device=self.device)
        self.image_offset = self.max_structural_entity_index + 1
        n_valid_images = 0
        n_valid_texts = 0

        raw_triplets = []

        # --- Relation: (item_i, hasImage, image_i) ---
        if self.v_feat is not None:
            self.image_relation_index = self.max_structural_relation_index + 1
            has_v_mask = self.v_feat.abs().sum(dim=1) > 0
            valid_items_v = item_ids[has_v_mask]
            valid_images = torch.arange(len(self.v_feat), device=self.device)[has_v_mask] + self.image_offset
            n_valid_images = len(valid_images)

            if len(valid_items_v) > 0:
                has_image = torch.stack([
                    valid_items_v,
                    torch.full_like(valid_items_v,  self.image_relation_index),
                    valid_images
                ], dim=1)
                raw_triplets.extend([[str(x) for x in triplet] for triplet in has_image.tolist()])

        self.text_offset  = self.image_offset + n_valid_images
        # --- Relation: (item_i, hasText, text_i) ---
        if self.t_feat is not None:
            self.text_relation_index = self.max_structural_relation_index + 2
            has_t_mask = self.t_feat.abs().sum(dim=1) > 0
            
            valid_items_t = item_ids[has_t_mask]
            valid_texts = torch.arange(len(self.t_feat), device=self.device)[has_t_mask] + self.text_offset
            n_valid_texts = len(valid_texts)
        
            if len(valid_items_t) > 0:
                has_text = torch.stack([
                    valid_items_t,
                    torch.full_like(valid_items_t, self.text_relation_index), 
                    valid_texts
                ], dim=1)
                raw_triplets.extend([[str(x) for x in triplet] for triplet in has_text.tolist()])

        triplets = Triplets()
        for raw_triplet in raw_triplets:
            triplets.add(raw_triplet[0], raw_triplet[1], raw_triplet[2])

        return triplets, n_valid_images, n_valid_texts

 
    # ──────────────────────────────────────────────────────────────────────────
    # MKG Attention Layer
    # ──────────────────────────────────────────────────────────────────────────
 
    def mkg_attention_layer(self, head_emb, users=False):
        triple_embs = []
        h_indices = []

        if users:
            triplets = self.triplets.data
        else:
            triplets = self.triplets.get_entity_triplets()

        for triplet in triplets:
            if users:
                h = head_emb[self.head_to_idx_with_users[triplet.head]]
            else:
                h = head_emb[self.head_to_idx[triplet.head]]
            r = self.get_relation_embedding(triplet.relation)
            if triplet.tail in self.head_to_idx:
                if users:
                    t = head_emb[self.head_to_idx_with_users[triplet.tail]]
                else:
                    t = head_emb[self.head_to_idx[triplet.tail]]
            else:
                t = self.get_entity_embedding(triplet.tail)

            # e(h,r,t) = W1 * [h || r || t]  — eq. 2
            e_hrt = self.W1(torch.cat([h, r, t], dim=-1))
            triple_embs.append(e_hrt)
            if users:
                h_indices.append(self.head_to_idx_with_users[triplet.head])
            else:
                h_indices.append(self.head_to_idx[triplet.head])

        triple_embs = torch.stack(triple_embs, dim=0)            # [n_triplets, dim]
        h_idx       = torch.tensor(h_indices, device=self.device) # [n_triplets]

        # π̃(h,r,t) = LeakyReLU(W2 * e(h,r,t))  — eq. 3
        attn_logits = F.leaky_relu(self.W2(triple_embs)).squeeze(-1)  # [n_triplets]

        attn = self.scatter_softmax(attn_logits, h_idx, num_nodes=head_emb.shape[0])

        # e_agg(h) = Σ_{(r,t)∈N(h)} π(h,r,t) * e(h,r,t)  — eq. 1
        e_agg = torch.zeros_like(head_emb)
        for i in range(len(h_idx)):
            e_agg[h_idx[i]] += attn[i] * triple_embs[i]

        # aggregation layer  — eq. 5 / eq. 6
        if self.aggregation == "add":
            new_head_emb = self.W3(head_emb) + e_agg
        else:
            new_head_emb = self.W3(torch.cat([head_emb, e_agg], dim=-1))

        return new_head_emb  
    
    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────
 
    def forward_kg(self):
        # ── 1. ENTITY ENCODER ─────────────────────────────────────────────────
        entities_head_emb = self._get_multimodal_head_embeddings()
 
        # ── 2. KG EMBEDDING MODULE ────────────────────────────────────────────
        # Propagate over the multi-modal KG to learn knowledge-aware entity emb
        for _ in range(self.n_layers):
            entities_head_emb = self.mkg_attention_layer(entities_head_emb)
 
        return entities_head_emb
     
    def forward_rec(self, head_emb=None):

        if head_emb is None:
            head_emb = self.forward_kg()

        if self.last_head_index_before_user is None:
            self.user_ids = sorted(self.triplets.get_unique_users())
            self.last_head_index_before_user = max(self.head_to_idx.values())
            self.head_to_idx_with_users = self.head_to_idx.copy()
            user_index = self.last_head_index_before_user
            for user_id in self.user_ids:
                user_index = user_index + 1
                self.head_to_idx_with_users[user_id] = user_index
        for user_id in self.user_ids:       
            head_emb.append(self.get_entity_embedding(user_id))
            

        all_head_emb = torch.stack(head_emb, dim=0)  # [n_users+n_items, dim]

        if self.head_indexes_items is None:
            self.head_indexes_items = []
            for item_id in sorted(self.triplets.get_unique_items()):
                self.head_indexes_items.append(self.head_to_idx_with_users[item_id])


        layer_outputs_users = [all_head_emb[self.last_head_index_before_user + 1:]]
        layer_outputs_items = [all_head_emb[self.head_indexes_items]]

        for _ in range(self.n_layers):
            all_head_emb = self.mkg_attention_layer(all_head_emb, users=True)
            layer_outputs_users.append(all_head_emb[self.last_head_index_before_user + 1:])
            layer_outputs_items.append(all_head_emb[self.head_indexes_items])

        user_all_embeddings = torch.cat(layer_outputs_users, dim=1)
        item_all_embeddings = torch.cat(layer_outputs_items, dim=1)

        return user_all_embeddings, item_all_embeddings
    # ──────────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────────
 

    def compute_kg_loss(self, head_emb):
    
        # ── Creazione batch di triplet ─────────────────────────────────────
        all_triplets = list(self.triplets)
        if self.batch_size_triplets >= len(all_triplets):
            batch_triplets = all_triplets
        else:
            batch_triplets = random.sample(all_triplets, self.batch_size_triplets)

        # ── Raccolta embedding ─────────────────────────────────────────────
        h_emb     = []
        r_emb     = []
        t_emb     = []
        t_neg_emb = []

        for triplet in batch_triplets:
            
            h_emb.append(head_emb[self.head_to_idx[triplet.head]])
            r_emb.append(self.get_relation_embedding(triplet.relation))
            if triplet.tail in self.head_to_idx:
                t_emb.append(head_emb[self.head_to_idx[triplet.tail]])
            else:
                t_emb.append(self.get_entity_embedding(triplet.tail))

            # tail corrotta — type constrained per modalità
            t_neg_id = self._corrupt_tail_for_relation(triplet)
            if t_neg_id in self.head_to_idx:
                t_neg_emb.append(head_emb[self.head_to_idx[t_neg_id]])
            else:       
                t_neg_emb.append(self.get_entity_embedding(t_neg_id))

        h_emb     = torch.stack(h_emb,     dim=0)  # [batch, dim]
        r_emb     = torch.stack(r_emb,     dim=0)  # [batch, dim]
        t_emb     = torch.stack(t_emb,     dim=0)  # [batch, dim]
        t_neg_emb = torch.stack(t_neg_emb, dim=0)  # [batch, dim]

        # ── TransE scoring — eq. 7 ────────────────────────────────────────
        score_valid  = torch.norm(h_emb + r_emb - t_emb,     p=2, dim=-1)  # [batch]
        score_broken = torch.norm(h_emb + r_emb - t_neg_emb, p=2, dim=-1)  # [batch]

        # ── Pairwise ranking loss — eq. 8 ─────────────────────────────────
        loss_kg = -torch.log(torch.sigmoid(score_broken - score_valid)).mean()

        return loss_kg



    def _corrupt_tail_for_relation(self, triplet):
        valid_tails = set(
            t.tail for t in self.triplets
            if t.head == triplet.head and t.relation == triplet.relation
        )

        if self.v_feat is not None and triplet.relation == self.image_relation_index:
            range_min = self.image_offset
            range_max = self.image_offset + self.n_valid_images
        elif self.t_feat is not None and triplet.relation == self.text_relation_index:
            range_min = self.text_offset
            range_max = self.text_offset + self.n_valid_texts
        else:
            range_min = self.n_triplets_users
            range_max = self.n_structural_entities

        # campiona finché non trovi un tail non valido
        corrupted = random.randint(range_min, range_max - 1)
        while corrupted in valid_tails:
            corrupted = random.randint(range_min, range_max - 1)
        return corrupted
    

    def compute_rec_loss(self, interaction, user_all, item_all):
        user     = interaction[0]  # [batch_size] — ID utenti del batch
        pos_item = interaction[1]  # [batch_size] — ID item positivi
        neg_item = interaction[2]  # [batch_size] — ID item negativi

        # ── indicizzazione — prendi solo le righe del batch ───────────────
        u_emb  = user_all[user, :]      # [batch_size, dim]
        pi_emb = item_all[pos_item, :]  # [batch_size, dim]
        ni_emb = item_all[neg_item, :]  # [batch_size, dim]

        # ── BPR loss — eq. 11 ─────────────────────────────────────────────
        # ŷ(u,i) = e*_u · e*_i
        pos_scores = torch.mul(u_emb, pi_emb).sum(dim=1)  # [batch_size]
        neg_scores = torch.mul(u_emb, ni_emb).sum(dim=1)  # [batch_size]

        loss_bpr = self.mf_loss(pos_scores, neg_scores)

        # ── L2 Regularization ─────────────────────────────────────────────
        loss_reg = self.reg_weight * (
            self.user_embedding(user).norm(p=2).pow(2) +
            self.entity_embedding(pos_item).norm(p=2).pow(2) +
            self.entity_embedding(neg_item).norm(p=2).pow(2)
        )

        return loss_bpr + loss_reg

       
 
    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────
 
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_all_embeddings, item_all_embeddings = self.forward_rec()
        u_embeddings = user_all_embeddings[user, :]
 
        # dot with all item embeddings → ranking scores
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores
    
    
    def scatter_softmax(src, index, num_nodes=None): #attn_logits, h_idx, num_nodes=head_emb.shape[0])
        # 1. Calcoliamo il massimo per ogni gruppo per stabilità numerica (evita overflow)
        out_max = torch.zeros(num_nodes, device=src.device) if num_nodes else torch.zeros(int(index.max()) + 1, device=src.device)
        out_max.index_reduce_(0, index, src, reduce='amax', include_self=False)
        
        # 2. Esponenziale della differenza
        out = (src - out_max[index]).exp()
        
        # 3. Somma degli esponenziali per gruppo
        out_sum = torch.zeros_like(out_max)
        out_sum.index_add_(0, index, out)
        
        # 4. Normalizzazione
        return out / (out_sum[index] + 1e-16)