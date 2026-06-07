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
from utils.triplet import Triplet, Triplets
from collections import defaultdict


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
        self.modality_dropout_ratio = config['modality_dropout_ratio']
        self.kg_loss_weight = config['kg_loss_weight'] \
            if config['kg_loss_weight'] is not None else 1.0
        self.use_contrastive = config['use_contrastive'] \
            if config['use_contrastive'] is not None else False
        self.use_layernorm = config['use_layernorm'] \
            if config['use_layernorm'] is not None else True
        self.dropout_ratio = config['dropout_ratio'] if config['dropout_ratio'] is not None else 0.2

        # load triplets
        if config['model_enriched_triples_format'] and config['dataset_support_triplets']:
            self.triplets = dataset.kg_triplets()
        else:
            raise Exception("The model cannot be trained!")
        
        # Initialization
        self._init_model()
        self._initialize_idxs()
        self.lean_dataset = len(dataset)
        self.train_batches = self._initialize_batches_kg(self.lean_dataset)
        self.n_train_batches = len(self.train_batches)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────


    def _shuffle_triplets(self, triplets):
        """
        Shuffles triplets by grouping image and text relations of the same item (head)
        together, ensuring they stay adjacent before batch slicing.
        """
        mm_groups = defaultdict(list)
        other_triplets = []

        for triplet in triplets:  
            h, r = triplet.head, triplet.relation
            if int(r) == self.image_relation_index or int(r) == self.text_relation_index:
                mm_groups[h].append(triplet)
            else:
                other_triplets.append(triplet)
           
        master_groups = list(mm_groups.values())
        for triplet in other_triplets:
            master_groups.append([triplet])

        random.shuffle(master_groups)

        shuffled_triplets = [triplet for group in master_groups for triplet in group]

        return shuffled_triplets



    def _initialize_batches_kg(self, maximum_n_batch, min_triplets=10):
        """
        Partition the Knowledge Graph triplets into batches for training.

        This method prepares the triplet data for the Knowledge Graph embedding module. 
        It splits the total set of triplets into a specified number of batches, 
        ensuring a balanced distribution while respecting minimum size constraints 
        to maintain training stability.

        The partitioning logic ensures that:
        1. Each batch contains at least 'min_triplets'.
        2. The total number of batches does not exceed 'maximum_n_batch'.

        Args:
            maximum_n_batch (int): The maximum number of batches to create.
            min_triplets (int, optional): The minimum number of triplets required 
                per batch. Defaults to 10.

        Returns:
            bathces: a list containing all the created batches.
        """
        all_triplets = self.triplets.get_entity_triplets().copy()
        all_triplets = self._shuffle_triplets(all_triplets)
        
        total = len(all_triplets)
        
        max_feasible_batches = total // min_triplets
        
        if max_feasible_batches == 0:
            return [all_triplets]
        
        actual_n_batch = min(maximum_n_batch, max_feasible_batches)
        
        base_size = total // actual_n_batch
        remainder = total % actual_n_batch  
        
        batches = []
        start = 0
        for i in range(actual_n_batch):
            extra = 1 if i < remainder else 0
            end = start + base_size + extra
            batches.append(all_triplets[start:end])
            start = end
        
        return batches
 
    def _init_model(self):
        """
        Initialize all the weights and some instance attributes.
        """
        self.max_structural_entity_index = max(int(entity) for entity in self.triplets.get_unique_entities_and_users()) # The maximum index of structural entities (excluding multi-modal entities, which are appended later)
        self.max_structural_relation_index =  max(int(r) for r in self.triplets.get_unique_relations()) # The maximum index of relations except the multi-modal ones, which are appended later.

        self.n_structural_entities = len(self.triplets.get_unique_entities()) 
        self.entity_embedding = nn.Embedding(self.n_structural_entities, self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        

        self.n_base_relations = (self.max_structural_relation_index + 1) \
                 + (1 if self.v_feat is not None else 0) \
                 + (1 if self.t_feat is not None else 0)
        self.n_relations = 2 * self.n_base_relations
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
        self.user_ids = [str(x) for x in sorted(int(i) for i in self.triplets.get_unique_users())]
        self.n_users = len(self.user_ids) 
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.shared_structural_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
            nn.LayerNorm(self.embedding_dim) if self.use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.item_ids = [str(x) for x in sorted(int(i) for i in self.triplets.get_unique_items())]
        self.n_items = len(self.item_ids) 
        new_triplets, self.n_valid_images, self.n_valid_texts = self._build_modality_kg_triplets() 
        if self. n_valid_images is not None and self.n_valid_images > 0:
           self.image_fc = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.embedding_dim),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(self.dropout_ratio),
            )       
        if self.n_valid_texts is not None and self.n_valid_texts > 0:
           self.text_fc = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.embedding_dim),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(self.dropout_ratio),
            )
        
        self.triplets.extend(new_triplets)

        self.mf_loss = BPRLoss()

        self.W1 = nn.Sequential(nn.Linear(self.embedding_dim * 3, self.embedding_dim), nn.Dropout(self.dropout_ratio))  
        self.W2 = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Dropout(self.dropout_ratio))            
        self.W3 = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.Dropout(self.dropout_ratio))  

        self.valid_tails = {} # Used to create the corrupted tails
        for t in self.triplets.data:
            if t.relation != "0":
                key = (t.head, t.relation)
                if key not in self.valid_tails:
                    self.valid_tails[key] = set()
                self.valid_tails[key].add(t.tail)

    
    def _build_modality_kg_triplets(self):
        """
        Construct the triplets representing the associations between items 
        and their multimodal data (images and texts).

        Returns:
            triplets (list): the list of resulting triplets
            n_valid_images (int): number of valid images
            n_valid_texts (int): number of valid texts
        """
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

        # --- Relation: (item_i, hasText, text_i) ---
        self.text_offset  = self.image_offset + n_valid_images
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
        """
        Set up the propagation graph index tensors.

        Every graph node (user, item, KG entity, image, text) has a contiguous
        integer id in [0, n_nodes), so head_emb is indexed directly by entity id
        and no id->position mapping is needed. Bidirectional (KGAT-style)
        propagation is built unconditionally: for each (h, r, t) a reverse edge
        (t, r + n_base_relations, h) is added.
        """
        self.n_nodes = self.text_offset + self.n_valid_texts

        self.all_node_ids = torch.arange(self.n_nodes, dtype=torch.long, device=self.device)

        triplets = self.triplets.data
        h_indices = [int(t.head) for t in triplets]
        r_indices = [int(t.relation) for t in triplets]
        t_indices = [int(t.tail) for t in triplets]

        # KGAT-style reverse edges for every relation 
        rev_h = list(t_indices)
        rev_r = [r + self.n_base_relations for r in r_indices]
        rev_t = list(h_indices)
        h_indices += rev_h
        r_indices += rev_r
        t_indices += rev_t

        self.triplet_h_idx = torch.tensor(h_indices, dtype=torch.long, device=self.device)
        self.triplet_r_idx = torch.tensor(r_indices, dtype=torch.long, device=self.device)
        self.triplet_t_idx = torch.tensor(t_indices, dtype=torch.long, device=self.device)

        self.head_user_idx_tensor = torch.tensor([int(u) for u in self.user_ids], dtype=torch.long, device=self.device)
        self.head_item_idx_tensor = torch.tensor([int(i) for i in self.item_ids], dtype=torch.long, device=self.device)


    # ──────────────────────────────────────────────────────────────────────────
    # Utils
    # ──────────────────────────────────────────────────────────────────────────
    def _get_entity_embedding(self, entity):
        """
        Return the embedding vector for a given entity.
        Args:
            entity (str): the entity identifier.         
        """
        return self._get_entity_embeddings_batch([entity])[0]

    def _get_entity_embeddings_batch(self, entity_ids):
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

        user_mask    = entity_ids < self.n_users
        entity_mask  = (entity_ids >= self.n_users) & (entity_ids < self.image_offset)
        image_mask   = (entity_ids >= self.image_offset) & (entity_ids < self.text_offset)
        text_mask    = entity_ids >= self.text_offset

        if user_mask.any():
            idx = entity_ids[user_mask]
            out[user_mask] = self.user_projected_embeddings[idx]

        if entity_mask.any():
            idx = entity_ids[entity_mask] - self.n_users
            out[entity_mask] = self.entity_projected_embeddings[idx]

        if image_mask.any():
            idx = entity_ids[image_mask] - self.image_offset
            v = self.v_feat[idx].detach().clone()
            if self.training:
                drop_mask = torch.rand(v.shape[0], device=self.device) < self.modality_dropout_ratio
                v[drop_mask] = 0.0
            out[image_mask] = self.image_fc(v)


        if text_mask.any():
            idx = entity_ids[text_mask] - self.text_offset
            t = self.t_feat[idx].detach().clone()
            if self.training:
                drop_mask = torch.rand(t.shape[0], device=self.device) < self.modality_dropout_ratio
                t[drop_mask] = 0.0
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
    


    def _get_head_embeddings(self):
        """
        Return the embeddings of all head entities and users in the triplets.
        Returns:
            Tensor: matrix of shape (n_head_entities, embedding_dim)
        """
        return self._get_entity_embeddings_batch(self.all_node_ids)

      
    # ──────────────────────────────────────────────────────────────────────────
    # MKG Attention Layer
    # ──────────────────────────────────────────────────────────────────────────
 
    def _mkg_attention_layer(self, head_emb):
        """
        Perform one Multi-modal Knowledge Graph Attention layer forward pass.
        Args:
            head_emb (Tensor): matrix of shape (n_head_entities, embedding_dim)
                containing the current embeddings of all head entities
        Returns:
            Tensor: updated embeddings of shape (n_head_entities, embedding_dim)
        """
        h = head_emb[self.triplet_h_idx]
        r = self._get_relation_embedding_batch(self.triplet_r_idx) 
       
        t = head_emb[self.triplet_t_idx]   
      
        # e(h,r,t) = W1 * [h || r || t]  — eq. 2
        e_hrt = self.W1(torch.cat([h, r, t], dim=-1))
        
        # π̃(h,r,t) = LeakyReLU(W2 * e(h,r,t))  — eq. 3
        attn_logits = F.leaky_relu(self.W2(e_hrt), negative_slope=self.leaky_relu_slope).squeeze(-1)  # [n_triplets]

        attn = self._scatter_softmax(attn_logits, self.triplet_h_idx, head_emb.shape[0])

        # e_agg(h) = Σ_{(r,t)∈N(h)} π(h,r,t) * e(h,r,t)  — eq. 1
        e_agg = torch.zeros_like(head_emb)

        e_agg.scatter_add_(0, self.triplet_h_idx.unsqueeze(1).expand_as(e_hrt),
                    attn.unsqueeze(1) * e_hrt)
        
        new_head_emb = self.W3(torch.cat([head_emb, e_agg], dim=-1))
        #new_head_emb = head_emb + update
        #new_head_emb = self.layer_norm(new_head_emb)
        return new_head_emb  
    
    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward_kg(self):
        """
        Perform the forward pass of the Knowledge Graph Embedding module.
        User embeddings are excluded from the output.
        Returns:
            Tensor: matrix of shape (n_entities, embedding_dim)
                containing the updated embeddings of all head entities,
                excluding users.
        """
        self.entity_projected_embeddings = self.shared_structural_fc(
            self.entity_embedding.weight
        )
        self.relation_projected_embeddings = self.shared_structural_fc(
            self.relation_embedding.weight
        )
        self.user_projected_embeddings = self.shared_structural_fc(
            self.user_embedding.weight
        )
        head_embs = self._get_head_embeddings()
 
        # Propagate over the multi-modal KG to learn knowledge-aware entity emb
        for _ in range(self.n_layers):
            head_embs = self._mkg_attention_layer(head_embs)
        return head_embs
     

    def forward_rec(self):
        """
        Perform the forward pass of the Recommendation module.

        Returns:
            user_all_embeddings (Tensor): matrix of shape (n_users, embedding_dim * (n_layers + 1))
                containing the aggregated embeddings of all users across all layers.
            item_all_embeddings (Tensor): matrix of shape (n_items, embedding_dim * (n_layers + 1))
                containing the aggregated embeddings of all items across all layers.
        """
        self.entity_projected_embeddings = self.shared_structural_fc(
            self.entity_embedding.weight
        )
        self.relation_projected_embeddings = self.shared_structural_fc(
            self.relation_embedding.weight
        )
        self.user_projected_embeddings = self.shared_structural_fc(
            self.user_embedding.weight
        )

        all_head_emb = self._get_head_embeddings()

        layer_outputs_users = [all_head_emb[self.head_user_idx_tensor]]
        layer_outputs_items = [all_head_emb[self.head_item_idx_tensor]]

        for _ in range(self.n_layers):
            all_head_emb = self._mkg_attention_layer(all_head_emb)
            layer_outputs_users.append(all_head_emb[self.head_user_idx_tensor])
            layer_outputs_items.append(all_head_emb[self.head_item_idx_tensor])

        user_all_embeddings = torch.cat(layer_outputs_users, dim=1)
        item_all_embeddings = torch.cat(layer_outputs_items, dim=1)

        return user_all_embeddings, item_all_embeddings
    

    # ──────────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────────
 
    def compute_kg_loss(self, batch_index, head_emb=None):
        """
        Compute the Knowledge Graph pairwise ranking loss over a batch of triplets.
        It Encourages valid triplets to have a lower score than corrupted ones.
        Args:
            batch_index (int): index of the current recommendation module batch.
            head_emb (Tensor, optional): precomputed head entity embeddings. 
                If None, they are computed.

        Returns:
            loss_kg: scalar value of the KG loss for the batch.
        """
        if batch_index < self.n_train_batches:
            batch_triplets = self.train_batches[batch_index]
        else:
            batch_triplets = random.choice(self.train_batches)

        if head_emb is None:
            head_emb = self.forward_kg()
       

        batch_h_idx = torch.tensor([int(t.head) for t in batch_triplets], dtype=torch.long, device=self.device)
        batch_r_idx = torch.tensor([int(t.relation) for t in batch_triplets], dtype=torch.long, device=self.device)
        h_emb    = head_emb[batch_h_idx]
        r_emb    = self._get_relation_embedding_batch(batch_r_idx)

        batch_t_idx = torch.tensor([int(t.tail) for t in batch_triplets], dtype=torch.long, device=self.device)
        t_emb = head_emb[batch_t_idx]

        t_neg_idx = torch.tensor([int(self._corrupt_tail_for_relation(t)) for t in batch_triplets], dtype=torch.long, device=self.device)
        t_neg_emb = head_emb[t_neg_idx]

        # ── TransE scoring — eq. 7 ────────────────────────────────────────
        score_valid  = (h_emb + r_emb - t_emb).pow(2).sum(dim=-1) 
        score_broken = (h_emb + r_emb - t_neg_emb).pow(2).sum(dim=-1)  

        # ── Pairwise ranking loss — eq. 8 ─────────────────────────────────
        loss_kg = self.kg_loss_weight * F.softplus(score_valid - score_broken).mean()


        # ── Modality contrastive loss (optional) ──────────────────────────
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

                    # sort by head id so image[i] and text[i] refer to the same item
                    img_order = torch.argsort(img_heads[img_sel])
                    txt_order = torch.argsort(txt_heads[txt_sel])

                    v_rows = img_rows[img_sel][img_order]
                    t_rows = txt_rows[txt_sel][txt_order]

                    z_v = F.normalize(self.image_fc(self.v_feat[v_rows]), dim=-1)
                    z_t = F.normalize(self.text_fc(self.t_feat[t_rows]), dim=-1)

                    logits = torch.matmul(z_v, z_t.T) / 0.07
                    labels = torch.arange(len(z_v), device=self.device)

                    loss_cl = (F.cross_entropy(logits, labels) +
                               F.cross_entropy(logits.T, labels)) / 2.0
                    return loss_kg + 0.2 * loss_cl
        return loss_kg



    def _corrupt_tail_for_relation(self, triplet):
        """
        Given a triplet, it returns a corrupted tail entity.
        If the relation associates an item with its image or text,
        the corrupted tail is sampled from the same modality (another image or text).
        Otherwise, the corrupted tail is sampled randomly from all entities.
        Args:
            triplet (Triplet): the input triplet.
        Returns:
            corrupted (str): the identifier of the corrupted tail entity.
        """
        if self.v_feat is not None and int(triplet.relation) == self.image_relation_index:
            range_min = self.image_offset
            range_max = self.image_offset + self.n_valid_images - 1
        elif self.t_feat is not None and int(triplet.relation) == self.text_relation_index:
            range_min = self.text_offset
            range_max = self.text_offset + self.n_valid_texts - 1
        else:
            range_min = self.n_users
            range_max = self.max_structural_entity_index

        valid_tails = self.valid_tails[(triplet.head, triplet.relation)]
        corrupted = str(random.randint(range_min, range_max))
        attempts = 0
        while corrupted in valid_tails and attempts < 100:
            corrupted = str(random.randint(range_min, range_max))
            attempts += 1
        return corrupted
    

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

        # ── BPR loss — eq. 11 ─────────────────────────────────────────────
        pos_scores = torch.mul(u_emb, pi_emb).sum(dim=1)  
        neg_scores = torch.mul(u_emb, ni_emb).sum(dim=1) 

        loss_bpr = self.mf_loss(pos_scores, neg_scores)

        u_base_emb = self.user_embedding(user)
        pi_base_emb = self.entity_embedding(pos_item)
        ni_base_emb = self.entity_embedding(neg_item)

        batch_size = u_base_emb.shape[0]
        reg_loss = self.reg_weight * (u_base_emb.pow(2).sum()
                                      + pi_base_emb.pow(2).sum()
                                      + ni_base_emb.pow(2).sum()) / batch_size

        return loss_bpr + reg_loss


       
 
    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────
 
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
            src (Tensor): input logits of shape (n_triplets,)
            index (Tensor): group assignments of shape (n_triplets,),
                where each value indicates the head entity the triplet belongs to
            num_nodes (int): total number of nodes.
        Returns:
            Tensor: normalized attention weights of shape (n_triplets,).        
        """
        out_max = torch.zeros(num_nodes, device=src.device)
        out_max.index_reduce_(0, index, src, reduce='amax', include_self=False)
        
        out = (src - out_max[index]).exp()
        
        out_sum = torch.zeros_like(out_max)
        out_sum.index_add_(0, index, out)
        
        return out / (out_sum[index] + 1e-16)
    

    def update_train_batches_kg(self):
        self.train_batches = self._initialize_batches_kg( self.lean_dataset)
        self.n_train_batches = len(self.train_batches)
        