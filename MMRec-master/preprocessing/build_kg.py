
import os
import ast
import csv
from collections import defaultdict, Counter

import pandas as pd

_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _norm_predicate(uri):
    """URI -> short lowercase name (last path/# segment). Unifies dbo/dbp variants."""
    return str(uri).rsplit('/', 1)[-1].split('#')[-1].lower()


# =============================================================================
# MovieLens KG builder (DBpedia 1-hop + native ML-1M genre), cleaned + enriched
# =============================================================================

# DBpedia predicate (normalised) -> relation name.
# NOTE: 'genre' is intentionally absent here: the genre is taken from the native
# MovieLens-1M genres (100% coverage), which are far richer than DBpedia's sparse one.
ML_PRED2REL = {
    'starring': 'starring', 'subject': 'subject', 'director': 'director',
    'producer': 'producer', 'distributor': 'distributor', 'editing': 'editing',
    'writer': 'writer', 'cinematography': 'cinematography',
    'musiccomposer': 'musiccomposer', 'language': 'language', 'country': 'country',
    'author': 'author', 'composer': 'composer', 'basedon': 'basedon',
    'productioncompany': 'studio', 'studio': 'studio',     # -> studio
    'screenplay': 'screenplay',
    'series': 'franchise', 'subsequentwork': 'franchise', 'previouswork': 'franchise',
}

# Fixed relation indices (0 reserved for interaction). Keeps triples.txt reproducible.
ML_RELATION_INDEX = {
    'starring': 1, 'subject': 2, 'director': 3, 'producer': 4, 'distributor': 5,
    'editing': 6, 'writer': 7, 'cinematography': 8, 'musiccomposer': 9, 'genre': 10,
    'language': 11, 'country': 12, 'author': 13, 'composer': 14, 'basedon': 15,
    'studio': 16, 'screenplay': 17, 'franchise': 18,
}


def _ensure_ml1m_movies(path):
    """Download MovieLens-1M movies.dat (which carries the genres) if not present."""
    if os.path.exists(path):
        return
    import urllib.request, zipfile, io
    print("downloading ml-1m movies.dat ...")
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    data = urllib.request.urlopen(
        urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    with z.open('ml-1m/movies.dat') as f, open(path, 'w', encoding='utf-8') as out:
        out.write(f.read().decode('latin-1'))


def construct_movielens_triples(dataset_name="movielens",
                                interact_file_name="movielens_1m.inter",
                                dbpedia_file="ml25m_dbpedia_1hop.tsv",
                                movies_dat="ml1m_movies.dat",
                                extended_mapping="ml1m_full_extended_mapping.tsv",
                                min_entity_degree=2,
                                subject_hub_threshold=500):
    """
    Build the cleaned + enriched MovieLens KG 'triples.txt' in a single pass:

      1. user-item interactions (relation 0);
      2. semantic relations from the DBpedia 1-hop dump (whitelisted in ML_PRED2REL),
         keeping only URI objects whose subject is a catalog item;
      3. native MovieLens-1M genres, used in place of the sparse DBpedia genre;
      4. extra relations recovered from the raw dump (production studio, screenplay,
         franchise links);
      5. leaf-entity pruning: drop KG entities linked to fewer than min_entity_degree
         items;
      6. mega-hub pruning: drop overly generic entities that link to more than
         subject_hub_threshold items (very broad categories that add only coarse,
         non-discriminative connectivity);
      7. remap to contiguous indices and save the relation/entity mappings.

    Encoding: user node = userID; item node = itemID + n_user; KG entity nodes come
    after the items. Relation 0 = interaction. Forward edges only (inverses are added
    at load time by the dataloader).

    Files expected in ../data/{dataset_name}/: i_id_mapping.csv (dburl,index),
    the .inter file, the raw DBpedia .tsv (subject/predicate/object), and
    ml1m_full_extended_mapping.tsv (movie_id -> dburl). movies.dat is downloaded
    automatically if missing.
    """
    d = os.path.join(_DATA_ROOT, dataset_name)
    out_path = os.path.join(d, "triples.txt")
    rel_map_path = os.path.join(d, "relation_mapping.tsv")
    ent_map_path = os.path.join(d, "entity_mapping.tsv")
    interact_path = os.path.join(d, interact_file_name)

    n_user = pd.read_csv(interact_path, sep='\t', usecols=['userID'])['userID'].nunique()

    items_id2index = {}
    with open(os.path.join(d, "i_id_mapping.csv"), 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    items_id2index[row[0].strip()] = int(row[1]) + n_user
                except ValueError:
                    pass
    n_items = len(items_id2index)

    # -- 1) interaction triples (relation 0) ----------------------------------
    interaction_triples = []
    with open(interact_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) >= 2:
                interaction_triples.append((int(p[0]), 0, int(p[1]) + n_user))

    # -- 2)+4) scan raw DBpedia: whitelisted + extra relations (no genre) ------
    candidates = []                  # (item_index, relation_name, entity_key)
    obj_items = defaultdict(set)     # entity_key -> set(item_index)
    subj_deg = defaultdict(int)      # subject object -> #items (for hub pruning)
    with open(os.path.join(d, dbpedia_file), 'r', encoding='utf-8') as f:
        f.readline()  # header: subject predicate object
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2]
            if s not in items_id2index or not o.startswith('http'):
                continue
            rel = ML_PRED2REL.get(_norm_predicate(p))
            if rel is None:                 # not whitelisted (genre, wikilink, type, ...)
                continue
            si = items_id2index[s]
            candidates.append((si, rel, o))
            obj_items[o].add(si)
            if rel == 'subject':
                subj_deg[o] += 1

    # -- 3) native MovieLens-1M genre (replaces the sparse DBpedia genre) ------
    _ensure_ml1m_movies(os.path.join(d, movies_dat))
    mid2gen = {}
    for l in open(os.path.join(d, movies_dat), encoding='utf-8'):
        f3 = l.rstrip('\n').split('::')
        if len(f3) >= 3:
            gs = [g for g in f3[2].split('|') if g and g != '(no genres listed)']
            if gs:
                mid2gen[f3[0]] = gs
    mid2url = {}
    with open(os.path.join(d, extended_mapping), encoding='utf-8') as f:
        next(f)
        for l in f:
            p = l.rstrip('\n').split('\t')
            if len(p) >= 2 and p[1]:
                mid2url[p[0]] = p[1].strip()
    for mid, gs in mid2gen.items():
        url = mid2url.get(mid)
        if url in items_id2index:
            si = items_id2index[url]
            for g in gs:
                key = 'genre::' + g
                candidates.append((si, 'genre', key))
                obj_items[key].add(si)

    # -- 5)+6) pruning: leaf entities + generic subject mega-hubs --------------
    hub_subjects = {o for o, deg in subj_deg.items() if deg > subject_hub_threshold}

    def keep(rel, o):
        if rel == 'subject' and o in hub_subjects:
            return False                              # drop generic mega-hub categories
        return (o in items_id2index) or (len(obj_items[o]) >= min_entity_degree)

    kept = [(si, rel, o) for (si, rel, o) in candidates if keep(rel, o)]

    # -- 7) assign indices (fixed relation map; items reuse their own node) ----
    entity_id2index = dict(items_id2index)
    next_ent = n_user + n_items
    used_rel = {}
    semantic_triples = []
    for si, rel, o in kept:
        ri = ML_RELATION_INDEX[rel]
        used_rel[rel] = ri
        if o in entity_id2index:
            oi = entity_id2index[o]
        else:
            oi = next_ent
            entity_id2index[o] = oi
            next_ent += 1
        semantic_triples.append((si, ri, oi))
    semantic_triples = list(dict.fromkeys(semantic_triples))   # dedup exact duplicates

    # -- write triples.txt (forward only; inverses added at load time) ---------
    with open(out_path, 'w', encoding='utf-8') as f:
        for tr in interaction_triples:
            f.write("%d\t%d\t%d\n" % tr)
        for tr in semantic_triples:
            f.write("%d\t%d\t%d\n" % tr)

    # -- save mappings ---------------------------------------------------------
    with open(rel_map_path, 'w', encoding='utf-8') as f:
        f.write("index\tname\n0\tinteraction\n")
        for rel, ri in sorted(used_rel.items(), key=lambda x: x[1]):
            f.write(f"{ri}\t{rel}\n")
    with open(ent_map_path, 'w', encoding='utf-8') as f:
        f.write("index\turi\n")
        for uri, idx in entity_id2index.items():
            if idx >= n_user + n_items:               # only KG entities, not items
                f.write(f"{idx}\t{uri}\n")

    # -- report ----------------------------------------------------------------
    relc = Counter(ri for _, ri, _ in semantic_triples)
    print("\n========== MOVIELENS KG (cleaned + enriched) ==========")
    print(f"users={n_user}  items={n_items}  KG_entities={next_ent - (n_user + n_items)}")
    print(f"interaction edges={len(interaction_triples)}  semantic edges={len(semantic_triples)}")
    print(f"pruned subject mega-hubs (> {subject_hub_threshold} items): {len(hub_subjects)}")
    print("relations: " + ", ".join(
        f"{r}={ML_RELATION_INDEX[r]}({relc[ML_RELATION_INDEX[r]]})"
        for r in sorted(used_rel, key=lambda x: ML_RELATION_INDEX[x])))
    print(f"saved: {out_path}, {rel_map_path}, {ent_map_path}")


# =============================================================================
# Amazon (JSON metadata) KG builder: skip empty fields, leaf-prune, save mappings
# =============================================================================
def construct_triples_json_clean(dataset_name, interact_file_name, metadata_id_field,
                                 flat_relations=('brand',),
                                 related_relations=('bought_together',),
                                 min_entity_degree=2):
    """
    Build a *cleaned* KG triples.txt from JSON (Amazon-style) metadata:
      - flat_relations:    top-level scalar fields (e.g. 'brand') -> one entity each;
      - related_relations: keys under 'related' (e.g. 'bought_together') -> lists of
                           item ASINs (item-item or item-entity edges);
      - missing/None/empty values are SKIPPED (no phantom shared entity);
      - leaf entities (KG nodes linked to < min_entity_degree items) are dropped,
        while item-valued tails (ASINs in the catalog) are always kept;
      - contiguous indices are reassigned and relation/entity mappings are saved.

    Encoding: user node = userID; item node = itemID + n_user; KG entity nodes
    start at n_user + n_items. Relation 0 = user-item interaction.
    """
    dataset_dir = os.path.join(_DATA_ROOT, dataset_name)
    output_path       = os.path.join(dataset_dir, "triples.txt")
    i_id_mapping      = os.path.join(dataset_dir, "i_id_mapping.csv")
    interact_path     = os.path.join(dataset_dir, interact_file_name)
    meta_path         = os.path.join(dataset_dir, "metadata.json")
    relation_map_path = os.path.join(dataset_dir, "relation_mapping.tsv")
    entity_map_path   = os.path.join(dataset_dir, "entity_mapping.tsv")

    n_user = pd.read_csv(interact_path, sep='\t', usecols=['userID'])['userID'].nunique()

    items_id2index = {}
    with open(i_id_mapping, 'r', encoding='utf-8') as f:
        f.readline()  # header
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                items_id2index[parts[0]] = int(parts[1]) + n_user
    n_items = len(items_id2index)

    # -- 1) interaction triples (relation 0) ----------------------------------
    interaction_triples = []
    with open(interact_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                interaction_triples.append([int(parts[0]), 0, int(parts[1]) + n_user])

    # -- 2) load metadata (Python-dict style -> ast.literal_eval) --------------
    meta = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = ast.literal_eval(line); meta[d[metadata_id_field]] = d
            except (ValueError, SyntaxError, KeyError):
                continue

    # -- 3) collect candidates, SKIPPING missing/None/empty (kills phantom hub) -
    candidates = []
    obj_items  = defaultdict(set)
    def add(item_index, rel, key):
        candidates.append((item_index, rel, key)); obj_items[key].add(item_index)

    for asin, item_index in items_id2index.items():
        d = meta.get(asin)
        if d is None:
            continue
        for rel in flat_relations:
            v = d.get(rel)
            if v is None or (isinstance(v, str) and v.strip() == ''):
                continue
            add(item_index, rel, f"{rel}::{v}")
        related = d.get('related', {})
        if isinstance(related, dict):
            for rel in related_relations:
                for a2 in (related.get(rel) or []):
                    if a2:
                        add(item_index, rel, a2)

    # -- 4) leaf-entity pruning (item-valued tails are always kept) ------------
    def keep(key):
        return (key in items_id2index) or (len(obj_items[key]) >= min_entity_degree)
    kept = [(si, rel, ek) for (si, rel, ek) in candidates if keep(ek)]

    # -- 5) assign contiguous indices ------------------------------------------
    relation_id2index = {}; next_rel = 1                # 0 reserved for interaction
    entity_id2index = dict(items_id2index); next_ent = n_user + n_items
    semantic_triples = []
    for si, rel, ek in kept:
        if rel not in relation_id2index:
            relation_id2index[rel] = next_rel; next_rel += 1
        ri = relation_id2index[rel]
        if ek in entity_id2index:
            oi = entity_id2index[ek]
        else:
            oi = next_ent; entity_id2index[ek] = oi; next_ent += 1
        semantic_triples.append([si, ri, oi])

    # -- 6) write triples.txt (forward only; inverses added at load time) ------
    with open(output_path, 'w', encoding='utf-8') as f:
        for tr in interaction_triples: f.write("\t".join(map(str, tr)) + "\n")
        for tr in semantic_triples:    f.write("\t".join(map(str, tr)) + "\n")

    # -- 7) save mappings ------------------------------------------------------
    with open(relation_map_path, 'w', encoding='utf-8') as f:
        f.write("index\tname\n0\tinteraction\n")
        for name, idx in sorted(relation_id2index.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")
    with open(entity_map_path, 'w', encoding='utf-8') as f:
        f.write("index\tkey\n")
        for key, idx in entity_id2index.items():
            if idx >= n_user + n_items:
                f.write(f"{idx}\t{key}\n")

    # -- report ----------------------------------------------------------------
    n_kg = next_ent - (n_user + n_items)
    print("\n========== CLEANED Amazon KG ==========")
    print(f"users={n_user}  items={n_items}  KG_entities={n_kg}")
    print(f"relations: 0=interaction + {dict(relation_id2index)}")
    print(f"interaction edges={len(interaction_triples)}  semantic edges={len(semantic_triples)} "
          f"(candidates before prune: {len(candidates)})")
    print(f"total edges={len(interaction_triples)+len(semantic_triples)}")
    print(f"saved: {output_path}, {relation_map_path}, {entity_map_path}")


# =============================================================================
# Amazon Baby KG builder: brand + bought_together (local 2014 metadata) enriched
# with category + material from the Amazon-Reviews-2023 metadata; leaf-pruned.
# =============================================================================

BABY_2023_META_URL = ('https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023'
                      '/resolve/main/raw/meta_categories/meta_Baby_Products.jsonl')

# Fixed relation indices for baby (0 reserved for interaction).
BABY_RELATION_INDEX = {
    'bought_together': 1, 'brand': 2, 'category': 3, 'material': 4,
}


def _ensure_baby_2023_matched(matched_path, catalog_asins):
    """
    Ensure baby_2023_matched.jsonl exists: the Amazon-Reviews-2023 'Baby_Products'
    metadata records whose parent_asin is in the catalog, keeping only the fields
    needed for enrichment. Built by streaming the (~3 GB) HF metadata once.
    """
    if os.path.exists(matched_path):
        return
    import json
    import urllib.request
    print("streaming Amazon-Reviews-2023 Baby_Products metadata (~3 GB, one-time) ...")
    req = urllib.request.Request(BABY_2023_META_URL, headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req, timeout=120)
    with open(matched_path, 'w', encoding='utf-8') as out:
        for raw in r:
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            if rec.get('parent_asin') in catalog_asins:
                out.write(json.dumps({'parent_asin': rec.get('parent_asin'),
                                      'categories': rec.get('categories'),
                                      'details': rec.get('details')},
                                     ensure_ascii=False) + '\n')


def construct_baby_triples(dataset_name="baby",
                           interact_file_name="baby.inter",
                           metadata_id_field="asin",
                           min_entity_degree=2,
                           use_2023_enrichment=True):
    """
    Build the cleaned + enriched Amazon Baby KG 'triples.txt' in a single pass:

      1. user-item interactions (relation 0);
      2. brand + bought_together from the 2014 metadata.json;
      3. category (hierarchical) + material from the Amazon-Reviews-2023 metadata;
      4. leaf-entity pruning: drop KG entities linked to < min_entity_degree items;
         item-valued tails (bought_together ASINs in the catalog) are always kept;
      5. fixed relation indices + saved relation/entity mappings.

    Encoding: user = userID; item = itemID + n_user; KG entities after the items.
    Relation 0 = interaction. Forward edges only (inverses added at load time).
    Set use_2023_enrichment=False to build the base KG (brand + bought_together only).
    """
    d = os.path.join(_DATA_ROOT, dataset_name)
    out_path     = os.path.join(d, "triples.txt")
    interact_path = os.path.join(d, interact_file_name)
    meta_path    = os.path.join(d, "metadata.json")
    matched_path = os.path.join(d, "baby_2023_matched.jsonl")
    rel_map_path = os.path.join(d, "relation_mapping.tsv")
    ent_map_path = os.path.join(d, "entity_mapping.tsv")

    n_user = pd.read_csv(interact_path, sep='\t', usecols=['userID'])['userID'].nunique()

    # item ASIN -> node index
    items_id2index = {}
    with open(os.path.join(d, "i_id_mapping.csv"), 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                items_id2index[parts[0]] = int(parts[1]) + n_user
    n_items = len(items_id2index)

    # -- 1) interaction triples (relation 0) ----------------------------------
    interaction_triples = []
    with open(interact_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                interaction_triples.append((int(parts[0]), 0, int(parts[1]) + n_user))

    candidates = []                  # (item_index, relation_name, entity_key)
    obj_items = defaultdict(set)

    def add(item_index, rel, key):
        candidates.append((item_index, rel, key)); obj_items[key].add(item_index)

    # -- 2) brand + bought_together from local 2014 metadata.json --------------
    meta = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = ast.literal_eval(line); meta[rec[metadata_id_field]] = rec
            except (ValueError, SyntaxError, KeyError):
                continue
    for asin, item_index in items_id2index.items():
        rec = meta.get(asin)
        if rec is None:
            continue
        b = rec.get('brand')
        if b is not None and not (isinstance(b, str) and b.strip() == ''):
            add(item_index, 'brand', f"brand::{b}")
        related = rec.get('related', {})
        if isinstance(related, dict):
            for a2 in (related.get('bought_together') or []):
                if a2:
                    add(item_index, 'bought_together', a2)

    # -- 3) category + material from Amazon-Reviews-2023 metadata --------------
    if use_2023_enrichment:
        _ensure_baby_2023_matched(matched_path, set(items_id2index))
        import json
        for line in open(matched_path, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            item_index = items_id2index.get(rec.get('parent_asin'))
            if item_index is None:
                continue
            for level in (rec.get('categories') or [])[1:]:    # skip generic root
                if level and level.strip():
                    add(item_index, 'category', 'cat::' + level.strip())
            det = rec.get('details') or {}
            if isinstance(det, dict):
                mats = set()
                for k in ('Material Type', 'Material'):
                    v = det.get(k)
                    if v and isinstance(v, str) and v.strip():
                        mats.add(v.strip())
                for m in mats:
                    add(item_index, 'material', 'mat::' + m)

    # -- 4) leaf-entity pruning (item-valued tails always kept) ----------------
    def keep(key):
        return (key in items_id2index) or (len(obj_items[key]) >= min_entity_degree)
    kept = [(si, rel, k) for (si, rel, k) in candidates if keep(k)]

    # -- 5) assign indices (fixed relation map; items reuse their node) --------
    entity_id2index = dict(items_id2index)
    next_ent = n_user + n_items
    used_rel = {}
    semantic_triples = []
    for si, rel, k in kept:
        ri = BABY_RELATION_INDEX[rel]
        used_rel[rel] = ri
        if k in entity_id2index:
            oi = entity_id2index[k]
        else:
            oi = next_ent
            entity_id2index[k] = oi
            next_ent += 1
        semantic_triples.append((si, ri, oi))
    semantic_triples = list(dict.fromkeys(semantic_triples))   # dedup exact duplicates

    # -- write triples.txt -----------------------------------------------------
    with open(out_path, 'w', encoding='utf-8') as f:
        for tr in interaction_triples:
            f.write("%d\t%d\t%d\n" % tr)
        for tr in semantic_triples:
            f.write("%d\t%d\t%d\n" % tr)

    # -- save mappings ---------------------------------------------------------
    with open(rel_map_path, 'w', encoding='utf-8') as f:
        f.write("index\tname\n0\tinteraction\n")
        for rel, ri in sorted(used_rel.items(), key=lambda x: x[1]):
            f.write(f"{ri}\t{rel}\n")
    with open(ent_map_path, 'w', encoding='utf-8') as f:
        f.write("index\tkey\n")
        for key, idx in entity_id2index.items():
            if idx >= n_user + n_items:
                f.write(f"{idx}\t{key}\n")

    # -- report ----------------------------------------------------------------
    relc = Counter(ri for _, ri, _ in semantic_triples)
    print("\n========== BABY KG (cleaned + 2023-enriched) ==========")
    print(f"users={n_user}  items={n_items}  KG_entities={next_ent - (n_user + n_items)}")
    print(f"interaction edges={len(interaction_triples)}  semantic edges={len(semantic_triples)}")
    print("relations: " + ", ".join(
        f"{r}={BABY_RELATION_INDEX[r]}({relc[BABY_RELATION_INDEX[r]]})"
        for r in sorted(used_rel, key=lambda x: BABY_RELATION_INDEX[x])))
    print(f"saved: {out_path}, {rel_map_path}, {ent_map_path}")


# -- entry point --------------------------------------------------------------

construct_baby_triples("baby", "baby.inter", "asin",
                      min_entity_degree=2, use_2023_enrichment=True)

#construct_movielens_triples("movielens", "movielens_1m.inter", "ml25m_dbpedia_1hop.tsv",
#                            min_entity_degree=2, subject_hub_threshold=500)
