
import os
import json
import ast
import csv
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def construct_triplets_json(dataset_name, interact_file_name, metadata_id_field, metadata_relations_names):
    """
    Constructs a Knowledge Graph (KG) in triplet format (head, relation, tail)
    by combining user-item interaction triplets with semantic triplets extracted
    from a JSON metadata file.

    User-item interactions are treated as triplets of the form:
        (user_id, interaction, item_id)
    where item indices are offset by the number of users to avoid ID collisions
    between users and items in the entity space.

    Semantic triplets are extracted by navigating the JSON metadata structure
    using slash-separated paths (e.g., 'related/also_bought'), allowing nested
    fields to be used as relational predicates. Each (item, relation, entity)
    triplet is assigned contiguous integer indices. If the object entity is
    already known (either as an item or a previously seen entity), its existing
    index is reused; otherwise, a new index is assigned.

    The resulting triplets are saved to a file named 'triplets.txt' in the
    dataset directory.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'baby'). Used to locate
            the dataset directory at '../data/{dataset_name}'.
        interact_file_name (str): Name of the .inter file containing user-item
            interactions, expected to have at least 'userID' and 'itemID' columns
            separated by tabs.
        metadata_id_field (str): The key in the JSON metadata file used as the
            unique identifier for each item (e.g., 'asin').
        metadata_relations_names (list of str): A list of keys (or slash-separated
            nested paths) to be extracted from the metadata as relational predicates
            (e.g., ['brand', 'related/also_bought']).
        
    """

    relations = {}
    relations[0] = "Interact relation"
    index = 1
    for relation_name in metadata_relations_names:
        relations[index] = relation_name
        index = index + 1    

    dataset_dir = f"../data/{dataset_name}"
    output_path = os.path.join(dataset_dir, "triplets.txt")
    i_id_mapping = os.path.join(dataset_dir, "i_id_mapping.csv")
    u_id_mapping = os.path.join(dataset_dir, "u_id_mapping.csv")
    interact_path = os.path.join(dataset_dir, interact_file_name)

    meta_path = os.path.join(dataset_dir, "metadata.json")
    users_index2id = {}
    with open(u_id_mapping, 'r') as f:
        header = f.readline() 
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                id = parts[0]
                index = int(parts[1]) 
                users_index2id[index] = id
    n_user = len(users_index2id.keys())
    items_index2id = {}
    with open(i_id_mapping, 'r') as f:
        header = f.readline() 
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                id = parts[0] 
                index = int(parts[1]) + n_user
                items_index2id[index] = id
    n_items = len(items_index2id.keys())
    interaction_triplets = []
    with open(interact_path, 'r') as f:
        header = f.readline() 
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                user_id = int(parts[0]) # Head
                item_id = int(parts[1]) + n_user # Tail  
                relation = 0       
                print(f"New triple: {user_id} {relation} {item_id}")     
                interaction_triplets.append([user_id, relation, item_id])
    
    print("\n....Writing the interaction triplets\n\n")
    with open(output_path, 'w', encoding='utf-8') as f:
        for triple in interaction_triplets:
            line = "\t".join(map(str, triple))
            f.write(line + "\n")
    

    id_lookup = {}  
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = ast.literal_eval(line)
                id_lookup[data[f'{metadata_id_field}']] = data
            except (json.JSONDecodeError, KeyError):
                continue 

    entities_index2id = {}
    for item_index, item_id in items_index2id.items():
        additional_triplets = []
        for relation_name in metadata_relations_names:
            relation_index_found = None
            for relation_index, relation in relations.items():
                if relation == relation_name:
                    relation_index_found = relation_index
                    break
            relation_path = relation_name.split("/")
            current_level = id_lookup[item_id]
            for sub_path in relation_path:
                if isinstance(current_level, dict) and sub_path in current_level:
                    current_level = current_level[sub_path]
                else:
                    current_level = None
                    break 
            entities = current_level
            if not isinstance(entities, list):
                entities = [entities]
            for entity in entities:
                if entity in items_index2id.values():
                    index_found = None
                    for key, value in items_index2id.items():
                        if value == entity:
                            index_found = key
                            break
                    print(f"New triple: {item_index} {relation_index_found} {index_found}")     
                    additional_triplets.append([item_index, relation_index_found, index_found])
                else:
                    index_found = None
                    for index_entity, entity_id in entities_index2id.items():
                        if entity_id == entity:
                            index_found = index_entity
                            break
                    if index_found is not None:
                        print(f"New triple: {item_index} {relation_index_found} {index_found}")     
                        additional_triplets.append([item_index, relation_index_found, index_found])
                    else:
                        if entities_index2id:
                            new_index  = max(entities_index2id.keys()) + 1
                        else:
                            new_index = max(items_index2id.keys()) + 1
                        entities_index2id[new_index] = entity
                        print(f"New triple: {item_index} {relation_index_found} {new_index}")     
                        additional_triplets.append([item_index, relation_index_found, new_index])
        print("\n....Writing the additional triplets\n\n")
        with open(output_path, 'a', encoding='utf-8') as f:
            for triple in additional_triplets:
                line = "\t".join(map(str, triple))
                f.write(line + "\n")
            
# construct_triplets_json("baby", "baby.inter", "asin", ["brand", "related/also_bought"])




def construct_triplets(dataset_name, interact_file_name, additional_triplets_filename):
    """
    Constructs a Knowledge Graph (KG) in triplet format (head, relation, tail)
    by combining user-item interaction triplets with semantic triplets from an
    external knowledge source (e.g., DBpedia, Wikidata).

    User-item interactions are treated as triplets of the form:
        (user_id, interaction, item_id)
    where item indices are offset by the number of users to avoid ID collisions
    between users and items in the entity space.

    Semantic triplets (subject, predicate, object) are filtered to retain only
    those whose subject belongs to the set of known items and whose object is a
    URI (i.e., non-literal values such as plain text or numbers are discarded,
    as textual content is already encoded as dense multimodal features).

    All entities and relations are remapped to contiguous integer indices.
    The resulting triplets are saved to a file named 'triplets.txt' in the
    dataset directory.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'movielens'). Used to
            locate the dataset directory at '../data/{dataset_name}'.
        interact_file_name (str): Name of the .inter file containing user-item
            interactions, expected to have at least 'userID' and 'itemID' columns
            separated by tabs.
        additional_triplets_filename (str): Name of the .tsv file containing
            semantic triplets in (subject, predicate, object) format, with no
            header and tab-separated columns.
    """

    dataset_dir = f"../data/{dataset_name}"
    output_path = os.path.join(dataset_dir, "triplets.txt")
    i_id_mapping = os.path.join(dataset_dir, "i_id_mapping.csv")
    interact_path = os.path.join(dataset_dir, interact_file_name)
    additional_triplets_path = os.path.join(dataset_dir, additional_triplets_filename)
    n_user = pd.read_csv(interact_path, sep='\t', usecols=['userID'])['userID'].nunique()
    items_id2index = {}
    with open(i_id_mapping, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for parts in reader:
                if len(parts) >= 2:
                    id = parts[0] 
                    index = int(parts[1]) + n_user
                    items_id2index[id] = index
    n_items = len(items_id2index.keys())
    interaction_triplets = []
    with open(interact_path, 'r') as f:
        header = f.readline() 
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                user_id = int(parts[0]) # Head
                item_id = int(parts[1]) + n_user # Tail  
                relation = 0       
                print(f"New triple: {user_id} {relation} {item_id}")     
                interaction_triplets.append([user_id, relation, item_id])
    
    print("\n....Writing the interaction triplets\n\n")
    with open(output_path, 'w', encoding='utf-8') as f:
        for triple in interaction_triplets:
            line = "\t".join(map(str, triple))
            f.write(line + "\n")
    
    relation_id2index = {}
    entities_id2index = {}
    entities_id2index.update(items_id2index)
    relation_id2index["interaction"] = 0
    chunks = pd.read_csv(additional_triplets_path, sep='\t', chunksize=20000)
    for chunk in tqdm(chunks, desc="Processed chunk"):
        additional_triplets = []
        for _, row in chunk.iterrows():
            subject   = row["subject"]
            predicate = row["predicate"]
            obj       = row["object"]

            if subject not in items_id2index.keys():
                continue 
            if not str(obj).startswith('http'):
                continue

            if predicate not in relation_id2index:
                max_index_until_now = max(relation_id2index.values())
                relation_id2index[predicate] = max_index_until_now + 1
            predicate_index =  relation_id2index[predicate]

            subject_index =  entities_id2index[subject]

            if obj not in entities_id2index:
                max_index_until_now = max(entities_id2index.values())
                entities_id2index[obj] = max_index_until_now +1
            object_index =  entities_id2index[obj]

            print(f"New triple: {subject_index} {predicate_index} {object_index}")     
            additional_triplets.append([subject_index, predicate_index, object_index])
        print("\n....Writing the additional triplets\n")
        with open(output_path, 'a', encoding='utf-8') as f:
            for triple in additional_triplets:
                line = "\t".join(map(str, triple))
                f.write(line + "\n")
            
#construct_triplets("movielens", "movielens_1m.inter", "ml25m_dbpedia_1hop.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# Cleaned KG builder: predicate whitelist + leaf-entity removal + saved mappings
# ─────────────────────────────────────────────────────────────────────────────

# Semantic predicates worth keeping (short name, lowercase). dbo:/dbp: variants
# of the same concept (e.g. ontology/starring vs property/starring) collapse to
# the same name → the relation is unified instead of duplicated.
PREDICATE_WHITELIST = {
    'subject',        # dct:subject  → Wikipedia categories (films of 1995, ...)
    'starring',       # actors
    'director',
    'writer',
    'producer',
    'musiccomposer',
    'composer',
    'cinematography',
    'editing',
    'distributor',
    'genre',
    'country',
    'language',
    'basedon',
    'series',
    'author',
}


def _norm_predicate(uri):
    """URI → short lowercase name (last path/# segment). Unifies dbo/dbp variants."""
    return str(uri).rsplit('/', 1)[-1].split('#')[-1].lower()


def construct_triplets_clean(dataset_name, interact_file_name, additional_triplets_filename,
                             predicate_whitelist=PREDICATE_WHITELIST, min_entity_degree=2):
    """
    Build a *cleaned* KG triplets.txt:
      - keep only whitelisted (semantic) predicates, unifying dbo/dbp variants;
      - drop leaf entities (KG entities linked to < min_entity_degree items),
        since they connect nothing and only add noise;
      - reassign contiguous indices and SAVE the relation/entity mappings.

    Encoding: user node = userID; item node = itemID + n_user; KG entity nodes
    start right after the items. Relation 0 = user-item interaction.
    """
    dataset_dir = f"../data/{dataset_name}"
    output_path        = os.path.join(dataset_dir, "triplets.txt")
    i_id_mapping       = os.path.join(dataset_dir, "i_id_mapping.csv")
    interact_path      = os.path.join(dataset_dir, interact_file_name)
    additional_path    = os.path.join(dataset_dir, additional_triplets_filename)
    relation_map_path  = os.path.join(dataset_dir, "relation_mapping.tsv")
    entity_map_path    = os.path.join(dataset_dir, "entity_mapping.tsv")

    n_user = pd.read_csv(interact_path, sep='\t', usecols=['userID'])['userID'].nunique()

    # item URI -> node index
    items_id2index = {}
    with open(i_id_mapping, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for parts in reader:
            if len(parts) >= 2:
                items_id2index[parts[0]] = int(parts[1]) + n_user
    n_items = len(items_id2index)

    # ── 1) interaction triplets (relation 0) ──────────────────────────────────
    interaction_triplets = []
    with open(interact_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                interaction_triplets.append([int(parts[0]), 0, int(parts[1]) + n_user])

    # ── 2) scan DBpedia, keep only whitelisted semantic triples ───────────────
    candidates = []                       # (subject_index, pred_name, object_uri)
    obj_items = defaultdict(set)          # object_uri -> set of item indices
    for chunk in tqdm(pd.read_csv(additional_path, sep='\t', chunksize=200000),
                      desc="Scanning DBpedia"):
        for s, p, o in zip(chunk['subject'], chunk['predicate'], chunk['object']):
            if s not in items_id2index:
                continue
            o = str(o)
            if not o.startswith('http'):
                continue
            pn = _norm_predicate(p)
            if pn not in predicate_whitelist:
                continue
            si = items_id2index[s]
            candidates.append((si, pn, o))
            obj_items[o].add(si)

    # ── 3) drop leaf entities (item-objects are always kept) ──────────────────
    def keep(o):
        return (o in items_id2index) or (len(obj_items[o]) >= min_entity_degree)
    kept = [(si, pn, o) for (si, pn, o) in candidates if keep(o)]

    # ── 4) assign contiguous indices ──────────────────────────────────────────
    relation_id2index = {}
    next_rel = 1                          # 0 reserved for interaction
    entity_id2index = dict(items_id2index)
    next_ent = n_user + n_items
    semantic_triplets = []
    for si, pn, o in kept:
        if pn not in relation_id2index:
            relation_id2index[pn] = next_rel
            next_rel += 1
        ri = relation_id2index[pn]
        if o in entity_id2index:
            oi = entity_id2index[o]
        else:
            oi = next_ent
            entity_id2index[o] = oi
            next_ent += 1
        semantic_triplets.append([si, ri, oi])

    # ── 5) write triplets.txt (forward only; inverses added at load time) ─────
    with open(output_path, 'w', encoding='utf-8') as f:
        for tr in interaction_triplets:
            f.write("\t".join(map(str, tr)) + "\n")
        for tr in semantic_triplets:
            f.write("\t".join(map(str, tr)) + "\n")

    # ── 6) save mappings ──────────────────────────────────────────────────────
    with open(relation_map_path, 'w', encoding='utf-8') as f:
        f.write("index\tname\n0\tinteraction\n")
        for name, idx in sorted(relation_id2index.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")
    with open(entity_map_path, 'w', encoding='utf-8') as f:
        f.write("index\turi\n")
        for uri, idx in entity_id2index.items():
            if idx >= n_user + n_items:          # only KG entities, not items
                f.write(f"{idx}\t{uri}\n")

    # ── report ────────────────────────────────────────────────────────────────
    n_kg_entities = next_ent - (n_user + n_items)
    print("\n========== CLEANED KG ==========")
    print(f"users:            {n_user}")
    print(f"items:            {n_items}")
    print(f"KG entities kept:  {n_kg_entities}")
    print(f"relations kept:    {len(relation_id2index)}  -> {sorted(relation_id2index, key=relation_id2index.get)}")
    print(f"interaction edges: {len(interaction_triplets)}")
    print(f"semantic edges:    {len(semantic_triplets)}  (candidates before leaf-drop: {len(candidates)})")
    print(f"total edges:       {len(interaction_triplets) + len(semantic_triplets)}")
    print(f"saved: {output_path}, {relation_map_path}, {entity_map_path}")


construct_triplets_clean("movielens", "movielens_1m.inter", "ml25m_dbpedia_1hop.tsv",
                         min_entity_degree=2)




        




        

