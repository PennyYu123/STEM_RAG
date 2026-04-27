
import json
import torch
import pickle
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
import numpy as np
from rapidfuzz import process, fuzz
from torch_geometric.data import Data


def create_ggs_aware_scorer(GGS: nx.Graph, boost_factor: float = 1.5):
    
    def ggs_scorer(query: str, choice: str, score_cutoff=None) -> int:
        base_score = fuzz.WRatio(query, choice)

        if choice in GGS.nodes():
            enhanced_score = int(base_score * boost_factor)
            return enhanced_score
        return base_score
    
    return ggs_scorer

@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class Relation:
    """Represents a relation between entities."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Relation):
            return False
        return self.id == other.id


@dataclass
class Triple:
    """Represents a knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    
    def to_text(self) -> str:
        """Convert triple to text representation for embedding."""
        return f"{self.subject}->{self.predicate}->{self.object}"
    
    def __str__(self) -> str:
        return self.to_text()


class KnowledgeGraph:
    def __init__(self, triples):
      
        self.graph = nx.MultiDiGraph()
        self._build_graph(triples)
        self.nodes_list = list(self.graph.nodes())

    def _build_graph(self, triples):
  
        for head, relation, tail in triples:
            self.graph.add_edge(head, tail, relation=relation)

    def get_one_hop_neighbors(self, entity):

        if entity not in self.graph:
            raise RuntimeError
        
        neighbors = []
        
        for _, target, attr in self.graph.out_edges(entity, data=True):
            neighbors.append((
                target,
                attr['relation'],
                entity
            ))

        for source, _, attr in self.graph.in_edges(entity, data=True):
            neighbors.append((
                entity,
                attr['relation'],
                source
            ))
            
        return neighbors

    def find_similiar_entities(self, entity, embedding_dict=None):
        result = process.extractOne(entity, self.nodes_list, scorer=fuzz.WRatio)
        if result:
            match_str, score, index = result
            return {
                "best_match": match_str,
                "score": score,
                "index": index
            }
        return None

    def find_similiar_entities_with_bias(self, entity, GGS, embedding_dict=None):
        ggs_scorer = create_ggs_aware_scorer(GGS, boost_factor=1.5)
        result = process.extractOne(entity, self.nodes_list, scorer=ggs_scorer)
        if result:
            match_str, score, index = result
            return {
                "best_match": match_str,
                "score": score,
                "index": index
            }
        return None

class KGConverter:
    def __init__(self, nx_graph, item, ent2ids, rel2ids, ids2ent, ids2rel, entity_vectors, relation_vectors, entity_ids):
        
        self.nx_graph = nx_graph
        self.item = item
        self.ent2ids = ent2ids
        self.rel2ids = rel2ids
        self.ids2ent = ids2ent
        self.ids2rel = ids2rel

        kg_rels = list(set([x[1] for x in item["graph"]]))

        ids_ent_kg = [idx for idx, x in enumerate([self.ids2ent[y] for y in range(len(self.ids2ent))]) if x in self.nx_graph.graph.nodes()]
        self.entities_list = [self.ids2ent[id] for id in ids_ent_kg] # Obtaining the total sample space used in calculating the loss during training
        ids_rel_kg = [idx for idx, x in enumerate([self.ids2rel[y] for y in range(len(self.ids2rel))]) if x in kg_rels]
        
        entity_vectors_kg = [x for idx, x in enumerate(entity_vectors) if idx in ids_ent_kg]
        relation_vectors_kg = [x for idx, x in enumerate(relation_vectors) if idx in ids_rel_kg]
        
        self.entity_vectors = torch.tensor(entity_vectors_kg, dtype=torch.float)
        self.relation_vectors = torch.tensor(relation_vectors_kg, dtype=torch.float)

        self.abs_rel_map_ent = dict(zip(ids_ent_kg, range(len(entity_vectors_kg))))
        self.rel_abs_map_ent = dict(zip(range(len(entity_vectors_kg)), ids_ent_kg))
        self.abs_rel_map_rel = dict(zip(ids_rel_kg, range(len(relation_vectors_kg))))

        self.entity_ids_relative = [ids_ent_kg.index(x) for x in entity_ids]
        
        self.num_entities = len(entity_vectors_kg)
        self.num_relations = len(relation_vectors_kg)
        assert self.entity_vectors.shape[0] == self.num_entities
        assert self.relation_vectors.shape[0] == self.num_relations
        self.convert_to_data()

    def convert_to_data(self):

        triples = []
        for u, v, data in self.nx_graph.graph.edges(data=True):
            if 'relation' not in data:
                continue
            h = self.abs_rel_map_ent[self.ent2ids[u]]
            t = self.abs_rel_map_ent[self.ent2ids[v]]
            r = self.abs_rel_map_rel[self.rel2ids[data['relation']]]
            triples.append((h, r, t))
        
        triples = np.array(triples)
        edge_index = torch.tensor(triples[:, [0, 2]].T, dtype=torch.long)
        edge_type = torch.tensor(triples[:, 1], dtype=torch.long)
        
        self.data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=self.num_entities,
            num_relations=self.num_relations,
            rel_emb=self.relation_vectors
        )