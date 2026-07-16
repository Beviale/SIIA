# MKGAT — Multi-modal Knowledge Graph Attention Network (unofficial implementation)

This repository provides an unofficial PyTorch implementation of **MKGAT**
(*Multi-modal Knowledge Graphs for Recommender Systems*, Sun et al., CIKM 2020),
which has no official public implementation, together with **KGAT** and the **BPR**
baseline, all integrated into the [MMRec](https://github.com/enoche/MMRec) framework.

The goal is a fair comparative study of knowledge-aware and multi-modal
recommendation: BPR (collaborative signal only), KGAT (knowledge graph, no
multimodality), and MKGAT (knowledge graph + visual and textual features).

## Main features
- MKGAT with multi-modal entity encoders, MKG attention layers, and alternating
  training (recommendation phase + TransE-style KG embedding phase).
- Modality-aware negative sampling for the KG phase: corrupted tails are drawn
  from the same modality pool as the positive tail (image → image, text → text).
- Knowledge graph construction scripts for two datasets:
  - **Amazon Baby**: KG built from Amazon product metadata (brand, category, …).
  - **MovieLens-1M**: KG built from DBpedia triples plus native ML-1M genres,
    with hub pruning and namespace deduplication.
- Evaluation via the standard MMRec protocol: full ranking over the whole item
  catalog with Recall@K, NDCG@K, Precision@K, and MAP@K.
