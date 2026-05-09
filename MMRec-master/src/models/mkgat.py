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
        self.leaky_relu_slope = config['leaky_relu_slope']
        self.dropout_ratio = config['dropout_ratio']
        self.modality_dropout_ratio = 0.1

        # load dataset info
        if config['model_enriched_triples_format'] and config['dataset_support_triplets']:
            self.triplets = dataset.kg_triplets()
        else:
            raise Exception("The model cannot be trained!")
        self._init_model()
        self._initialize_idxs()
        self.train_batches = self._initialize_batch_kg(len(dataset))
        self.n_train_batches = len(self.train_batches)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────

    def _initialize_batch_kg(self, n_batch):
        all_triplets = self.entity_triplets.copy()
        random.shuffle(all_triplets)
        
        total = len(all_triplets)
        MIN_TRIPLETS = 10
        
        max_feasible_batches = total // MIN_TRIPLETS
        
        if max_feasible_batches == 0:
            return [all_triplets]
        
        actual_n_batch = min(n_batch, max_feasible_batches)
        
        # Distribuisce il resto tra i primi batch
        base_size = total // actual_n_batch
        remainder = total % actual_n_batch  # primi `remainder` batch avranno +1
        
        batches = []
        start = 0
        for i in range(actual_n_batch):
            extra = 1 if i < remainder else 0
            end = start + base_size + extra
            batches.append(all_triplets[start:end])
            start = end
        
        return batches
 
    def _init_model(self):
        self.max_structural_entity_index = max(int(entity) for entity in self.triplets.get_unique_entities_and_users()) 
        self.max_structural_relation_index =  max(int(r) for r in self.triplets.get_unique_relations())

        self.n_structural_entities = len(self.triplets.get_unique_entities()) 
        self.entity_embedding = nn.Embedding(self.n_structural_entities, self.embedding_dim)
        

        self.n_relations = (self.max_structural_relation_index + 1) \
                 + (1 if self.v_feat is not None else 0) \
                 + (1 if self.t_feat is not None else 0)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_dim)
    
        self.user_ids = sorted(self.triplets.get_unique_users())
        self.n_users = len(self.user_ids) 
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)

        self.shared_structural_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(p=self.dropout_ratio)
        )
        self.item_ids = sorted(self.triplets.get_unique_items())
        new_triplets, n_valid_images, n_valid_texts = self._build_kg_triplets()
        if n_valid_images is not None and n_valid_images > 0:
           self.image_fc = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.embedding_dim),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(p=self.dropout_ratio)
            )       
        if n_valid_texts is not None and n_valid_texts > 0:
           self.text_fc = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.embedding_dim),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(p=self.dropout_ratio)
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
            if t.relation != "0":
                key = (t.head, t.relation)
                if key not in self.valid_tails:
                    self.valid_tails[key] = set()
                self.valid_tails[key].add(t.tail)


    def _get_entity_embedding(self, entity):
        return self._get_entity_embeddings_batch([entity])[0]

    def _get_entity_embeddings_batch(self, entity_ids):
        entity_ids = torch.tensor([int(x) for x in entity_ids], dtype=torch.long, device=self.device)
                      
        out = torch.zeros(len(entity_ids), self.embedding_dim, device=self.device)

        user_mask    = entity_ids < self.n_users
        entity_mask  = (entity_ids >= self.n_users) & (entity_ids < self.image_offset)
        image_mask   = (entity_ids >= self.image_offset) & (entity_ids < self.text_offset)
        text_mask    = entity_ids >= self.text_offset

        if user_mask.any():
            idx = entity_ids[user_mask]
            out[user_mask] = self.shared_structural_fc(self.user_embedding(idx))

        if entity_mask.any():
            idx = entity_ids[entity_mask] - self.n_users
            out[entity_mask] = self.shared_structural_fc(self.entity_embedding(idx))

        if image_mask.any():
            idx = entity_ids[image_mask] - self.image_offset
            v = self.v_feat[idx].detach()
            if self.training:
                drop_mask = torch.rand(v.shape[0], device=self.device) < self.modality_dropout_ratio
                v[drop_mask] = 0.0
            out[image_mask] = self.image_fc(v)


        if text_mask.any():
            idx = entity_ids[text_mask] - self.text_offset
            t = self.t_feat[idx].detach()
            if self.training:
                drop_mask = torch.rand(t.shape[0], device=self.device) < self.modality_dropout_ratio
                t[drop_mask] = 0.0
            out[text_mask] = self.text_fc(t)

        return out  
    

    def _get_relation_embedding_batch(self, relation_ids):
        relation_ids_int = torch.tensor([int(x) for x in relation_ids], dtype=torch.long, device=self.device)
        return self.shared_structural_fc(self.relation_embedding(relation_ids_int))
    


    def _get_head_embeddings(self):
        entities_emb = []
        entities_emb.append(self._get_entity_embeddings_batch(self.head_entities))  
        return torch.cat(entities_emb, dim=0)
    
  
    def _build_kg_triplets(self):
        item_ids = torch.tensor([int(x) for x in self.item_ids], dtype=torch.long, device=self.device)
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
    


    def _initialize_idxs(self):
        head_to_idx = {}
        index = 0
        self.head_entities = sorted(self.triplets.get_all_head_entities())
        for entity in self.head_entities:
            head_to_idx[entity] = index
            index = index + 1
        self.head_to_idx = head_to_idx

        self.head_to_idx_with_users = self.head_to_idx.copy()
        self.last_head_index_before_user =len(self.head_to_idx) - 1
        user_index = self.last_head_index_before_user
        for user_id in self.user_ids:
            user_index = user_index + 1
            self.head_to_idx_with_users[user_id] = user_index
            

        h_indices, r_indices, t_is_head, t_head_indices, t_entity_indices = [], [], [], [], []
        for triplet in self.triplets.data:
            h_indices.append(self.head_to_idx_with_users[triplet.head])
            r_indices.append(int(triplet.relation))
            if triplet.tail in self.head_to_idx_with_users:
                t_is_head.append(True)
                t_head_indices.append(self.head_to_idx_with_users[triplet.tail])
                t_entity_indices.append(0)
            else:
                t_is_head.append(False)
                t_head_indices.append(0)
                t_entity_indices.append(int(triplet.tail))

        self.triplet_h_idx_with_u = torch.tensor(h_indices, dtype=torch.long, device=self.device)
        self.triplet_r_idx_with_u  = torch.tensor(r_indices, dtype=torch.long, device=self.device)
        self.triplet_t_is_head_with_u  = torch.tensor(t_is_head, device=self.device)
        self.triplet_t_head_idx_with_u = torch.tensor(t_head_indices, dtype=torch.long, device=self.device)
        self.triplet_t_ent_idx_with_u  = torch.tensor(t_entity_indices, dtype=torch.long, device=self.device)

        h_indices, r_indices, t_is_head, t_head_indices, t_entity_indices = [], [], [], [], []
        self.entity_triplets = self.triplets.get_entity_triplets()
        for triplet in self.entity_triplets:
            h_indices.append(self.head_to_idx[triplet.head])
            r_indices.append(int(triplet.relation))
            if triplet.tail in self.head_to_idx:
                t_is_head.append(True)
                t_head_indices.append(self.head_to_idx[triplet.tail])
                t_entity_indices.append(0)
            else:
                t_is_head.append(False)
                t_head_indices.append(0)
                t_entity_indices.append(int(triplet.tail))

        self.triplet_h_idx_without_u = torch.tensor(h_indices, dtype=torch.long, device=self.device)
        self.triplet_r_idx_without_u  = torch.tensor(r_indices, dtype=torch.long, device=self.device)
        self.triplet_t_is_head_without_u  = torch.tensor(t_is_head, device=self.device)
        self.triplet_t_head_idx_without_u = torch.tensor(t_head_indices, dtype=torch.long, device=self.device)
        self.triplet_t_ent_idx_without_u  = torch.tensor(t_entity_indices, dtype=torch.long, device=self.device)

 
    # ──────────────────────────────────────────────────────────────────────────
    # MKG Attention Layer
    # ──────────────────────────────────────────────────────────────────────────
 
    def _mkg_attention_layer(self, head_emb, users=False):
        if users:
            h = head_emb[self.triplet_h_idx_with_u]
            r = self.relation_embedding(self.triplet_r_idx_with_u)
        else:
            h = head_emb[self.triplet_h_idx_without_u]
            r = self._get_relation_embedding_batch(self.triplet_r_idx_without_u)
        if users:
            t_from_head   = head_emb[self.triplet_t_head_idx_with_u]
            t_from_entity = self._get_entity_embeddings_batch(self.triplet_t_ent_idx_with_u)
            t = torch.where(self.triplet_t_is_head_with_u.unsqueeze(1), t_from_head, t_from_entity)
        else:
            t_from_head   = head_emb[self.triplet_t_head_idx_without_u]
            t_from_entity = self._get_entity_embeddings_batch(self.triplet_t_ent_idx_without_u)
            t = torch.where(self.triplet_t_is_head_without_u.unsqueeze(1), t_from_head, t_from_entity)




        # e(h,r,t) = W1 * [h || r || t]  — eq. 2
        e_hrt = self.W1(torch.cat([h, r, t], dim=-1))
        
        # π̃(h,r,t) = LeakyReLU(W2 * e(h,r,t))  — eq. 3
        attn_logits = F.leaky_relu(self.W2(e_hrt)).squeeze(-1)  # [n_triplets]

        if users:
            attn = self._scatter_softmax(attn_logits, self.triplet_h_idx_with_u, head_emb.shape[0])
        else:
            attn = self._scatter_softmax(attn_logits, self.triplet_h_idx_without_u, head_emb.shape[0])


        # e_agg(h) = Σ_{(r,t)∈N(h)} π(h,r,t) * e(h,r,t)  — eq. 1
        e_agg = torch.zeros_like(head_emb)

        if users:
            e_agg.scatter_add_(0, self.triplet_h_idx_with_u.unsqueeze(1).expand_as(e_hrt),
                       attn.unsqueeze(1) * e_hrt)
        else:
            e_agg.scatter_add_(0, self.triplet_h_idx_without_u.unsqueeze(1).expand_as(e_hrt),
                       attn.unsqueeze(1) * e_hrt)


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
        entities_head_emb = self._get_head_embeddings()
 
        # ── 2. KG EMBEDDING MODULE ────────────────────────────────────────────
        # Propagate over the multi-modal KG to learn knowledge-aware entity emb
        for _ in range(self.n_layers):
            entities_head_emb = self._mkg_attention_layer(entities_head_emb)
 
        return entities_head_emb
     
    def forward_rec(self, head_emb=None):

        if head_emb is None:
            head_emb = self.forward_kg().detach()

        new_users_emb = self._get_entity_embeddings_batch(self.user_ids)
        all_head_emb = torch.cat([head_emb, new_users_emb], dim=0)
            
        if self.head_indexes_items is None:
            self.head_indexes_items = []
            for item_id in sorted(self.item_ids):
                self.head_indexes_items.append(self.head_to_idx_with_users[item_id])


        layer_outputs_users = [all_head_emb[self.last_head_index_before_user + 1:]]
        layer_outputs_items = [all_head_emb[self.head_indexes_items]]

        for _ in range(self.n_layers):
            all_head_emb = self._mkg_attention_layer(all_head_emb, users=True)
            layer_outputs_users.append(all_head_emb[self.last_head_index_before_user + 1:])
            layer_outputs_items.append(all_head_emb[self.head_indexes_items])

        user_all_embeddings = torch.cat(layer_outputs_users, dim=1)
        item_all_embeddings = torch.cat(layer_outputs_items, dim=1)

        return user_all_embeddings, item_all_embeddings
    # ──────────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────────
 

    def compute_kg_loss(self, batch_index, head_emb=None):
        if batch_index < self.n_train_batches:
            batch_triplets = self.train_batches[batch_index]
        else:
            batch_triplets = random.choice(self.batches)

        if head_emb is None:
            head_emb = self.forward_kg()
       

        h_idx    = torch.tensor([self.head_to_idx[t.head] for t in batch_triplets])
        h_emb    = head_emb[h_idx]  # un solo accesso, niente loop

        # Per r_emb
        r_ids    = torch.tensor([int(t.relation) for t in batch_triplets])
        r_emb    = self._get_relation_embedding_batch(r_ids)


        t_emb_list = []
        for t in batch_triplets:
            if t.tail in self.head_to_idx:
                t_emb_list.append(head_emb[self.head_to_idx[t.tail]])  # ← embedding, non indice
            else:
                t_emb_list.append(self._get_entity_embedding(t.tail))
        t_emb = torch.stack(t_emb_list, dim=0)
        del t_emb_list

        # t_neg_emb
        t_neg_emb_list = []
        for t in batch_triplets:
            t_neg_id = self._corrupt_tail_for_relation(t)
            if t_neg_id in self.head_to_idx:
                t_neg_emb_list.append(head_emb[self.head_to_idx[t_neg_id]])
            else:
                t_neg_emb_list.append(self._get_entity_embedding(t_neg_id))
        t_neg_emb = torch.stack(t_neg_emb_list, dim=0)
        del t_neg_emb_list


        # ── TransE scoring — eq. 7 ────────────────────────────────────────
        score_valid  = torch.norm(h_emb + r_emb - t_emb,     p=2, dim=-1)  # [batch]
        score_broken = torch.norm(h_emb + r_emb - t_neg_emb, p=2, dim=-1)  # [batch]

        # ── Pairwise ranking loss — eq. 8 ─────────────────────────────────
        loss_kg = -torch.log(torch.sigmoid(score_broken - score_valid)).mean()

        return loss_kg



    def _corrupt_tail_for_relation(self, triplet):

        if self.v_feat is not None and triplet.relation == self.image_relation_index:
            range_min = self.image_offset
            range_max = self.image_offset + self.n_valid_images
        elif self.t_feat is not None and triplet.relation == self.text_relation_index:
            range_min = self.text_offset
            range_max = self.text_offset + self.n_valid_texts
        else:
            range_min = self.n_users
            range_max = self.n_structural_entities

        corrupted = str(random.randint(range_min, range_max))
        valid_tails = self.valid_tails[(triplet.head, triplet.relation)]
        while corrupted in valid_tails:
            corrupted = random.randint(range_min, range_max)
        return corrupted
    

    def compute_rec_loss(self, interaction, user_all=None, item_all=None):

        if user_all is None or item_all is None:
            user_all, item_all = self.forward_rec()


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
 
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores
    
    
    def _scatter_softmax(self, src, index, num_nodes=None): #attn_logits, h_idx, num_nodes=head_emb.shape[0])
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