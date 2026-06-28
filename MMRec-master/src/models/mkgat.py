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
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss
from utils.triplet import Triplet, Triples
from collections import defaultdict

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class MKGAT(GeneralRecommender):
    r"""MKGAT is a multi-modal knowledge graph based recommender model.
 
    MKGAT introduces multi-modal entities (images and text) as first-class citizens
    of the knowledge graph. It proposes a multi-modal graph attention technique to
    conduct information propagation over MMKGs, and uses the resulting aggregated
    embedding for recommendation.
 
    We implement the model following the original author with a pairwise training mode.
    """
 
    def __init__(self, config, dataset, triples):
        super(MKGAT, self).__init__(config, dataset)

        # load configuration
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']  
        self.reg_weight = config['reg_weight'] 
        self.leaky_relu_slope = config['leaky_relu_slope']
        self.modality_dropout_ratio = config['modality_dropout_ratio']
        self.use_contrastive = config['use_contrastive'] \
            if config['use_contrastive'] is not None else False
        self.dropout_ratio = config['dropout_ratio'] if config['dropout_ratio'] is not None else 0.2
        self.message_dropout_ratio = config['message_dropout_ratio'] if config['message_dropout_ratio'] is not None else 0.1
        self.message_dropout = nn.Dropout(self.message_dropout_ratio)

        self.triples = triples

        # Initialization
        self._init_model()
        self._initialize_idxs()
 
    # __________________________________________________________________________
    # Initialization
    # __________________________________________________________________________
 
    def _init_model(self):
        """
        Initialize all the weights and some instance attributes.
        """
        self.max_structural_entity_index = max(int(entity) for entity in self.triples.get_unique_entities_and_users()) # The maximum index of structural entities (excluding multi-modal entities, which are appended later)
        self.max_structural_relation_index =  max(int(r) for r in self.triples.get_unique_relations()) # The maximum index of relations except the multi-modal ones, which are appended later.

        self.n_structural_item_and_entities = len(self.triples.get_unique_item_and_entities())
        self.user_ids = [str(x) for x in sorted(int(i) for i in self.triples.get_unique_users())]
        self.n_users = len(self.user_ids) 
        self.structural_node_embeddings = nn.Embedding(self.n_structural_item_and_entities +  self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.structural_node_embeddings.weight)
        

        self.n_relations = (self.max_structural_relation_index + 1) \
                 + (2 if self.v_feat is not None else 0) \
                 + (2 if self.t_feat is not None else 0)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    

        self.shared_structural_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(self.dropout_ratio),
        )

        self.item_ids = [str(x) for x in sorted(int(i) for i in self.triples.get_unique_items())]
        self.n_items = len(self.item_ids) 
        new_triples = self._build_modality_kg_triples() 
        self.image_fc = nn.Sequential(
            nn.Linear(self.v_feat.shape[1], self.embedding_dim),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(self.dropout_ratio),
        )     
        self.text_fc = nn.Sequential(
            nn.Linear(self.t_feat.shape[1], self.embedding_dim),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(self.dropout_ratio),
        )
        self.triples.extend(new_triples)

        self.mf_loss = BPRLoss()

        self.W1 = nn.Linear(self.embedding_dim * 3, self.embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.W1.weight)  
        self.W2 = nn.Linear(self.embedding_dim, 1,  bias=False)
        nn.init.xavier_uniform_(self.W2.weight)             
        self.W3 = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.W3.weight) 

    
    def _build_modality_kg_triples(self):
        """
        Construct the triples representing the associations between items 
        and their multimodal data (images and texts).

        Returns:
            triples (Triples): the resulting hasImage/hasText triples (both directions)
        """
        if self.v_feat is not None:
            assert len(self.item_ids) == len(self.v_feat), "item_ids e v_feat not aligned"
        if self.t_feat is not None:
            assert len(self.item_ids) == len(self.t_feat), "item_ids e t_feat not aligned"

        self.image_offset = self.max_structural_entity_index + 1

        raw_triples = []
        item_ids = torch.tensor([int(x) for x in self.item_ids], dtype=torch.long, device=self.device) 

        # --- Relation: (item_i, hasImage, image_i) ---
        if self.v_feat is not None:
            self.image_relation_index = self.max_structural_relation_index + 1
            image_relation_index_opp = self.image_relation_index + 1
            has_v_mask = self.v_feat.abs().sum(dim=1) > 0
            valid_items_v = item_ids[has_v_mask]
            valid_images = torch.arange(len(self.v_feat), device=self.device)[has_v_mask] + self.image_offset

            if len(valid_items_v) > 0:
                has_image = torch.stack([
                    valid_items_v,
                    torch.full_like(valid_items_v,  self.image_relation_index),
                    valid_images
                ], dim=1)
                raw_triples.extend([[str(x) for x in triplet] for triplet in has_image.tolist()])

        # --- Relation: (item_i, hasText, text_i) ---
        self.text_offset  = self.image_offset + len(self.v_feat)
        if self.t_feat is not None:
            self.text_relation_index = self.max_structural_relation_index + 3
            text_relation_index_opp = self.text_relation_index + 1
            has_t_mask = self.t_feat.abs().sum(dim=1) > 0
            valid_items_t = item_ids[has_t_mask]
            valid_texts = torch.arange(len(self.t_feat), device=self.device)[has_t_mask] + self.text_offset
        
            if len(valid_items_t) > 0:
                has_text = torch.stack([
                    valid_items_t,
                    torch.full_like(valid_items_t, self.text_relation_index), 
                    valid_texts
                ], dim=1)
                raw_triples.extend([[str(x) for x in triplet] for triplet in has_text.tolist()])

        triples = Triples()
        for raw_triplet in raw_triples:
            if int(raw_triplet[1]) == self.image_relation_index:
                triples.add(raw_triplet[0], raw_triplet[1], raw_triplet[2])
                triples.add(raw_triplet[2], image_relation_index_opp, raw_triplet[0])
            elif int(raw_triplet[1]) == self.text_relation_index:
                triples.add(raw_triplet[0], raw_triplet[1], raw_triplet[2])
                triples.add(raw_triplet[2], text_relation_index_opp, raw_triplet[0])

        return triples
    


    def _initialize_idxs(self):
        """
        Set up the propagation graph index tensors.
        """
        self.total_n_nodes = self.text_offset + len(self.t_feat)

        self.all_node_ids = torch.arange(self.total_n_nodes, dtype=torch.long, device=self.device)

        triples = self.triples.data
        h_indices = [int(t.head) for t in triples]
        r_indices = [int(t.relation) for t in triples]
        t_indices = [int(t.tail) for t in triples]

        self.triplet_h_idx = torch.tensor(h_indices, dtype=torch.long, device=self.device)
        self.triplet_r_idx = torch.tensor(r_indices, dtype=torch.long, device=self.device)
        self.triplet_t_idx = torch.tensor(t_indices, dtype=torch.long, device=self.device)

        self.head_user_idx_tensor = torch.tensor([int(u) for u in self.user_ids], dtype=torch.long, device=self.device)
        self.head_item_idx_tensor = torch.tensor([int(i) for i in self.item_ids], dtype=torch.long, device=self.device)


    # __________________________________________________________________________
    # Utils
    # __________________________________________________________________________
    def _get_node_embedding(self, entity):
        """
        Return the embedding vector for a given entity.
        Args:
            entity (str): the entity identifier.         
        """
        return self._get_node_embeddings_batch([entity])[0]

    def _get_node_embeddings_batch(self, entity_ids):
        """
        Return a matrix containing the embeddings of a batch of entities.
        Args:
            entity_ids (list of strings or tensor of integers): the list of entity identifiers.
        Returns:
            out: matrix of shape (n_entities, embedding_dim)
        """
        if not isinstance(entity_ids, torch.Tensor):
            entity_ids = torch.as_tensor([int(x) for x in entity_ids], dtype=torch.long, device=self.device)
        else:
            entity_ids = entity_ids.to(dtype=torch.long, device=self.device)
                      
        out = torch.zeros(len(entity_ids), self.embedding_dim, device=self.device)

        strctural_node_mask  = entity_ids < self.image_offset
        image_mask   = (entity_ids >= self.image_offset) & (entity_ids < self.text_offset)
        text_mask    = entity_ids >= self.text_offset

        if strctural_node_mask.any():
            idx = entity_ids[strctural_node_mask]
            out[strctural_node_mask] = self.node_projected_embeddings[idx]

        if image_mask.any():
            idx = entity_ids[image_mask] - self.image_offset
            v = F.normalize(self.v_feat[idx].detach(), dim=-1)
            if self.training and self.modality_dropout_ratio > 0:
                keep = (torch.rand(v.shape[0], 1, device=self.device) >= self.modality_dropout_ratio).to(v.dtype)
                v = v * keep
            out[image_mask] = self.image_fc(v)

        if text_mask.any():
            idx = entity_ids[text_mask] - self.text_offset
            t = F.normalize(self.t_feat[idx].detach(), dim=-1)
            if self.training and self.modality_dropout_ratio > 0:
                keep = (torch.rand(t.shape[0], 1, device=self.device) >= self.modality_dropout_ratio).to(t.dtype)
                t = t * keep
            out[text_mask] = self.text_fc(t)

        return out  
    


    def _get_relation_embedding_batch(self, relation_ids):
        """
        Return a matrix containing the embeddings of a batch of relations.
        Args:
            relation_ids (list of strings or tensor of integers): the list of relation identifiers.
        Returns:
            out: matrix of shape (n_relations, embedding_dim)
        """
        if isinstance(relation_ids, torch.Tensor):
            relation_ids_int = relation_ids.to(dtype=torch.long, device=self.device)
        else:
            relation_ids_int = torch.tensor([int(x) for x in relation_ids], dtype=torch.long, device=self.device)
        return self.relation_projected_embeddings[relation_ids_int]

      
    # __________________________________________________________________________
    # MKG Attention Layer
    # __________________________________________________________________________
 
    def _mkg_attention_layer(self, head_emb):
        """
        Perform an attention layer propagation.
        Args: 
            head embeddings
        Returns:
            new updated head embeddings
        """
        h = head_emb[self.triplet_h_idx]
        r = self._get_relation_embedding_batch(self.triplet_r_idx)
        t = head_emb[self.triplet_t_idx]

        e_hrt = self.W1(torch.cat([h, r, t], dim=-1))                                    
        attn_logits = F.leaky_relu(self.W2(e_hrt), negative_slope=self.leaky_relu_slope).squeeze(-1)  
        attn = self._scatter_softmax(attn_logits, self.triplet_h_idx, head_emb.shape[0])              

        e_agg = torch.zeros_like(head_emb)                                              
        e_agg.scatter_add_(0, self.triplet_h_idx.unsqueeze(1).expand_as(e_hrt),
                    attn.unsqueeze(1) * e_hrt)

        new_head_emb = self.W3(torch.cat([head_emb, e_agg], dim=-1))                     
        new_head_emb = self.message_dropout(new_head_emb)                               
        return new_head_emb
    
    # __________________________________________________________________________
    # Forward
    # __________________________________________________________________________

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.compute_rec_loss(*input)
        elif mode == 'train_kg':
            return self.compute_kg_loss(*input)
        elif mode == 'predict':
            return self.full_sort_predict(*input)
        else:
            raise NotImplementedError(f"forward: mode '{mode}' not managed")

    def forward_kg(self):
        """
        Perform the forward pass of the Knowledge Graph Embedding module.
        Returns:
            Tensor: matrix of shape (n_entities, embedding_dim)
                containing the updated embeddings of all head entities. 
        """
        self.node_projected_embeddings = self.shared_structural_fc(
            self.structural_node_embeddings.weight
        )
        self.relation_projected_embeddings = self.shared_structural_fc(
            self.relation_embedding.weight
        ) 

        node_embs = self._get_node_embeddings_batch(self.all_node_ids)

        # Propagate over the multi-modal KG to learn knowledge-aware entity emb
        for _ in range(self.n_layers):
            node_embs = self._mkg_attention_layer(node_embs)
        node_embs = F.normalize(node_embs, p=2, dim=-1)  
        return node_embs
     

    def forward_rec(self):
        """
        Perform the forward pass of the Recommendation module.

        Returns:
            user_all_embeddings (Tensor): matrix of shape (n_users, embedding_dim * (n_layers + 1))
                containing the aggregated embeddings of all users across all layers.
            item_all_embeddings (Tensor): matrix of shape (n_items, embedding_dim * (n_layers + 1))
                containing the aggregated embeddings of all items across all layers.
        """
        self.node_projected_embeddings = self.shared_structural_fc(
            self.structural_node_embeddings.weight
        )
        self.relation_projected_embeddings = self.shared_structural_fc(
            self.relation_embedding.weight
        ) 

        node_embs = self._get_node_embeddings_batch(self.all_node_ids)

        layer_outputs_users = [node_embs[self.head_user_idx_tensor]]
        layer_outputs_items = [node_embs[self.head_item_idx_tensor]]

        for _ in range(self.n_layers):
            node_embs = self._mkg_attention_layer(node_embs)
            node_embs_norm = F.normalize(node_embs, p=2, dim=-1)
            layer_outputs_users.append(node_embs_norm[self.head_user_idx_tensor])
            layer_outputs_items.append(node_embs_norm[self.head_item_idx_tensor])

        user_all_embeddings = torch.cat(layer_outputs_users, dim=1)
        item_all_embeddings = torch.cat(layer_outputs_items, dim=1)

        return user_all_embeddings, item_all_embeddings
    

    # __________________________________________________________________________
    # Loss
    # __________________________________________________________________________
 
    def compute_kg_loss(self, batch_h_idx, batch_r_idx, batch_t_idx, batch_t_neg_idx):
        """
        Compute the Knowledge Graph pairwise ranking loss over a batch of triples.
        It Encourages valid triples to have a lower score than corrupted ones.
         Args:
            batch_h_idx (tensor): tensor containing the heads
            batch_r_idx (tensor): tensor containing the relations
            batch_t_idx (tensor): tensor containing the tails
            batch_t_neg_idx (tensor): tensor containing the corripted tails
        Returns:
            loss_kg: scalar value of the KG loss for the batch.
        """
        node_embs = self.forward_kg()

        h_emb = node_embs[batch_h_idx]
        r_emb  = self._get_relation_embedding_batch(batch_r_idx)
        t_emb = node_embs[batch_t_idx]
        t_neg_emb = node_embs[batch_t_neg_idx]

        score_valid  = (h_emb + r_emb - t_emb).pow(2).sum(dim=-1) 
        score_broken = (h_emb + r_emb - t_neg_emb).pow(2).sum(dim=-1)  

        loss_kg = ((-1.0) * F.logsigmoid(score_broken - score_valid)).mean()

        # __ Modality contrastive loss (optional) __________________________
        if self.use_contrastive and self.v_feat is not None and self.t_feat is not None:
            image_mask = batch_r_idx == self.image_relation_index
            text_mask  = batch_r_idx == self.text_relation_index
            if image_mask.any() and text_mask.any():
                img_heads = batch_h_idx[image_mask]
                txt_heads = batch_h_idx[text_mask]
                img_rows  = batch_t_idx[image_mask] - self.image_offset   # row in v_feat
                txt_rows  = batch_t_idx[text_mask]  - self.text_offset    # row in t_feat

                common = set(img_heads.tolist()) & set(txt_heads.tolist())
                if len(common) >= 2:
                    common_t = torch.tensor(list(common), device=self.device)

                    img_sel = torch.isin(img_heads, common_t)
                    txt_sel = torch.isin(txt_heads, common_t)

                    img_order = torch.argsort(img_heads[img_sel])
                    txt_order = torch.argsort(txt_heads[txt_sel])

                    v_rows = img_rows[img_sel][img_order]
                    t_rows = txt_rows[txt_sel][txt_order]

                    z_v = self.image_fc(self.v_feat[v_rows])
                    z_t =self.text_fc(self.t_feat[t_rows])

                    logits = torch.matmul(z_v, z_t.T) / 0.07
                    labels = torch.arange(len(z_v), device=self.device)

                    loss_cl = (F.cross_entropy(logits, labels) +
                               F.cross_entropy(logits.T, labels)) / 2.0
                    loss_kg = loss_kg + 0.4 * loss_cl
        return loss_kg

    

    def compute_rec_loss(self, interaction, user_all=None, item_all=None):
        """
        Compute the recommendation loss for a batch of user-item interactions
        using Bayesian Personalized Ranking (BPR) loss with L2 regularization.

        Args:
            interaction (tuple): contains (user_ids, pos_item_ids, neg_item_ids)
            user_all (Tensor, optional): precomputed user embeddings of shape 
                (n_users, embedding_dim * (n_layers + 1)). If None, computed internally.
            item_all (Tensor, optional): precomputed item embeddings of shape
                (n_items, embedding_dim * (n_layers + 1)). If None, computed internally.

        Returns:
            loss: scalar value of the recommendation loss for the batch
        """
        if user_all is None or item_all is None:
            user_all, item_all = self.forward_rec()

        user     = interaction[0]       
        pos_item = interaction[1] 
        neg_item = interaction[2]                       

        u_emb  = user_all[user, :]      
        pi_emb = item_all[pos_item, :]  
        ni_emb = item_all[neg_item, :]  

        pos_scores = torch.mul(u_emb, pi_emb).sum(dim=1)  
        neg_scores = torch.mul(u_emb, ni_emb).sum(dim=1) 

        loss_bpr = self.mf_loss(pos_scores, neg_scores)

        u_base_emb = self.structural_node_embeddings(user)
        pi_base_emb = self.structural_node_embeddings(pos_item + self.n_users)
        ni_base_emb = self.structural_node_embeddings(neg_item + self.n_users)
        
        reg_loss = _L2_loss_mean(u_base_emb) + _L2_loss_mean(pi_base_emb) + _L2_loss_mean(ni_base_emb)
        return loss_bpr + (self.reg_weight * reg_loss)


       
 
    # __________________________________________________________________________
    # Inference
    # __________________________________________________________________________
 
    def full_sort_predict(self, interaction):
        """
        Compute the predicted scores for all items for a given batch of users.
        Used during evaluation to rank all items for each user.
        Args:
            interaction (tuple): contains the indices of the users to evaluate

        Returns:
            Tensor: score matrix of shape (n_users, n_items) where entry (u, i)
            is the predicted preference score of user u for item i
        """
        user = interaction[0]
        user_all_embeddings, item_all_embeddings = self.forward_rec()
        u_embeddings = user_all_embeddings[user, :]
 
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores
    

    def _scatter_softmax(self, src, index, num_nodes): 
        """
        Compute a numerically stable softmax.
        Args:
            src (Tensor): input logits of shape (n_triples,)
            index (Tensor): group assignments of shape (n_triples,),
                where each value indicates the head entity the triplet belongs to
            num_nodes (int): total number of nodes.
        Returns:
            Tensor: normalized attention weights of shape (n_triples,).        
        """
        out_max = torch.zeros(num_nodes, device=src.device)
        out_max.index_reduce_(0, index, src, reduce='amax', include_self=False)
        
        out = (src - out_max[index]).exp()
        
        out_sum = torch.zeros_like(out_max)
        out_sum.index_add_(0, index, out)
        
        return out / (out_sum[index] + 1e-16)
    
    
    
        