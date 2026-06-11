from dataclasses import dataclass
from typing import List, Set


@dataclass
class Triplet:
    """It represents a single KG triplet"""
    head: str
    relation: str
    tail: str

class Triplets:
    def __init__(self, triplets_list: List[Triplet] = None):
        self.data: List[Triplet] = triplets_list if triplets_list is not None else []

    def add(self, head: str, relation: str, tail: str):
        self.data.append(Triplet(head, relation, tail))

    def get_all_heads(self) -> List[str]:
        return [t.head for t in self.data]

    def get_all_tails(self) -> List[str]:
        return [t.tail for t in self.data]

    def get_all_relations(self) -> List[str]:
        return [t.relation for t in self.data]
    
    def get_all_unique_heads(self) -> Set[str]:
        return set(self.get_all_heads())

    def get_all_unique_tails(self) -> Set[str]:
        return set(self.get_all_tails())
    
    def get_unique_item_and_entities(self) -> Set[str]:
        entities = set()
        for t in self.data:
            if t.relation == "0":
                entities.add(t.tail)
            elif t.relation == "1":
                entities.add(t.head)
            else:
                entities.add(t.head)
                entities.add(t.tail)
        return entities
    
    def get_unique_users(self) -> Set[str]:
        users = set()
        for t in self.data:
            if t.relation == "0":
                users.add(t.head)
        return users
    
    def get_all_head_entities(self) -> Set[str]:
        heads = set()
        for t in self.data:
            if t.relation != '0' and t.relation != "1":
                heads.add(t.head)
        return heads

    def get_unique_items(self) -> Set[str]:
        items = set()
        for t in self.data:
            if t.relation == "0":
                items.add(t.tail)
        return items

    def get_unique_entities_and_users(self) -> Set[str]:
        entities = set()
        for t in self.data:
            entities.add(t.head)
            entities.add(t.tail)
        return entities
    
    def get_user_triplets(self) -> List[Triplet]:
        triplets_filtered = []
        for t in self.data:
            if t.relation == "0" or t.relation == "1":
                triplets_filtered.append(t)
        return triplets_filtered
    
    def get_entity_triplets(self) -> List[Triplet]:
        triplets_filtered = []
        for t in self.data:
            if t.relation != "0" and t.relation != "1":
                triplets_filtered.append(t)
        return triplets_filtered


    def get_unique_relations(self) -> Set[str]:
        return set(self.get_all_relations())
    

    def extend(self, triplets):
        if isinstance(triplets, Triplets):
            self.data.extend(triplets.data)
        elif isinstance(triplets, list):
            self.data.extend(triplets)
        else:
            raise TypeError(f"Expected Triplets or List[Triplet], got {type(triplets)}")


    def __len__(self):
        return len(self.data)