
import os
import json
import ast

def construct_triplets(dataset_name, interact_file_name, metadata_id_field, metadata_relations_names):
    """
        Constructs a Knowledge Graph in triplet format (head, relation, tail).
        This method integrates semantic Item-Entity relations from a JSON metadata file 
        with User-Item behavioral relations from an interaction (.inter) file. The 
        resulting graph merges domain knowledge with user activity into a unified structure.

        Upon completion, the generated triplets are saved in the dataset directory as triplets.txt.

        Args:
            dataset_name (str): The name of the dataset.
            interact_file_name (str): Path to the .inter file containing user-item 
                interactions.
            metadata_id_field (str): The key in the JSON file used as the unique 
                identifier for items.
            metadata_relations_names (list of str): A list of JSON keys to be 
                extracted as relational predicates (e.g., ['brand', 'category']).
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
            
construct_triplets("baby", "baby.inter", "asin", ["brand", "related/also_bought"])




        

