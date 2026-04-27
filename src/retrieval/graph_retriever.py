
import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict
import networkx as nx

from core.graph import KnowledgeGraph, Triple, Entity, Relation, KGConverter
from core.config import get_config
from sg.sg_builder import AllSchemaGraphs
from embedding.encoder import QwenEmbeddingEncoder
from storage.vector_store import VectorStore, create_vector_store
from tqdm import tqdm

logger = logging.getLogger(__name__)

def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask

class GraphRetriever:
   
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph = None,
        encoder: QwenEmbeddingEncoder = None,
        vector_store: Optional[VectorStore] = None,
        graph_retriever = None,
        config=None,
    ):

        self.kg = knowledge_graph
        self.encoder = encoder
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.config = config or get_config()
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        self.breadth_theta = 0.6

    
    def index_triples(self, all_triples: list = False) -> None:

        logger.info("Building vector index...")
        
        # Encode all entities
        # entity_ids = []
        self.entities = []
        self.entity_vectors = []

        self.relation_vectors = []
        self.relations = []

        for triple in all_triples:
            self.entities.append(triple[0])
            self.entities.append(triple[2])
            self.relations.append(triple[1])
        self.entities = list(set(self.entities))
        self.entity_ids = list(range(len(self.entities)))
        self.relations = list(set(self.relations))
        self.relation_ids = list(range(len(self.relations)))

        self.entities = self.entities
        self.entity_ids = self.entity_ids
        self.relations = self.relations
        self.relation_ids = self.relation_ids

        self.ent2ids = dict(zip(self.entities, self.entity_ids))
        self.ids2ent = dict(zip(self.entity_ids, self.entities))
        self.rel2ids = dict(zip(self.relations, self.relation_ids))
        self.ids2rel = dict(zip(self.relation_ids, self.relations))

        # for entity_id, entity in self.kg.entities.items():
        for entity_id, entity in tqdm(zip(self.entity_ids, self.entities)):
            if entity_id in self._entity_embeddings and not force_rebuild:
                raise RuntimeError
                self.entity_vectors.append(self._entity_embeddings[entity_id])
            else:
                embedding = self.encoder.encode_entity(entity)
                self.entity_vectors.append(embedding)
                self._entity_embeddings[entity_id] = embedding
        
        # Add to vector store
        if self.entity_ids:
            self.vector_store.add_vectors(self.entity_ids, np.array(self.entity_vectors))
            logger.info(f"Added {len(self.entity_ids)} entity vectors to index")
        
        # for relation_id, relation in self.kg.relations.items():
        for relation_id, relation in tqdm(zip(self.relation_ids, self.relations)):
            embedding = self.encoder.encode_relation(relation)
            self.relation_vectors.append(embedding)

        # Add to vector store
        if self.relation_ids:
            self.vector_store.add_vectors(self.relation_ids, np.array(self.relation_vectors))
            logger.info(f"Added {len(self.relation_ids)} relation vectors to index")
        
        logger.info("Vector index built successfully")

    def match(self, entity_anchor, similar_entity, schema_graph, KG, GGS, S, reasoning_graph, strategy, last=None):
        schema_edges = schema_graph.get_one_hop_neighbors(entity_anchor)
        KG_edges = KG.get_one_hop_neighbors(similar_entity)
        # neighbors = [neighbor for neighbor in neighbors if not('m.' in neighbor[0] or 'g.' in neighbor[0] or 'm.' in neighbor[2] or 'g.' in neighbor[2])]
        KG_edges = list(set(KG_edges))
        encode_texts = [self.triple_serialize(KG_edge) for KG_edge in KG_edges]
        neighbor_embeds = self.encoder.encode_texts(encode_texts)

        for edge in schema_edges:
            if edge == last: continue
            if strategy == 'PRECISION':
                self.step_precision(entity_anchor, similar_entity, schema_graph, KG, GGS, S, \
                                        reasoning_graph, edge, KG_edges, neighbor_embeds, strategy)
            elif strategy == 'BREADTH':
                self.step_breadth(entity_anchor, similar_entity, schema_graph, KG, GGS, S, \
                                        reasoning_graph, edge, KG_edges, neighbor_embeds, strategy)
            else:
                raise RuntimeError
    
    def get_another(self, edge, entity):
        if edge[0] == entity:
            return edge[2]
        else:
            return edge[0]
    
    def triple_serialize(self, triple):
        return f"{triple[0]}_{triple[1]}_{triple[2]}"

    def edge_rectification(self, GGS, u, v):
        if GGS.has_edge(u, v):
            return 1/2
        else:
            return 0

    def step_precision(self, entity_anchor, similar_entity, schema_graph, KG, GGS, S, reasoning_graph, edge, KG_edges, neighbor_embeds, strategy):
        matched_edges = []
        for KG_edge in KG_edges:
            if edge[1] == KG_edge[1] and not edge in reasoning_graph:
                matched_edges.append(KG_edge)

        if not matched_edges:
            schema_edge_embed = self.encoder.encode_text(self.triple_serialize(edge))
            triple_bias = np.array([self.edge_rectification(GGS, KG_edge[0], KG_edge[2]) for KG_edge in KG_edges])

            sim_scores = (neighbor_embeds * schema_edge_embed).sum(-1) + triple_bias
            max_ids = sim_scores.argsort()
            id_pos = len(max_ids)-1
            idx = max_ids[id_pos]
            while True:
                if KG_edges[idx] in reasoning_graph:
                    id_pos -= 1
                    idx = max_ids[id_pos]
                else:
                    matched_edges.append(KG_edges[idx])
                    S+=sim_scores[idx]
                    break
        else:
            S+=1

        reasoning_graph.extend(matched_edges)
        entity_anchor_next = self.get_another(edge, entity_anchor)
        for matched_edge in matched_edges:
            similar_entity_next = self.get_another(matched_edge, similar_entity)
            self.match(entity_anchor_next, similar_entity_next, schema_graph, KG, GGS, S, reasoning_graph, strategy, last=edge)

    def step_breadth(self, entity_anchor, similar_entity, schema_graph, KG, GGS, S, reasoning_graph, edge, KG_edges, neighbor_embeds, strategy):
        matched_edges = []
        all_scores = []
        for KG_edge in KG_edges:
            if edge[1] == KG_edge[1] and not edge in reasoning_graph:
                matched_edges.append(KG_edge)
                all_scores.append(S+1)

        matched_edges = []
        if not matched_edges:
            schema_edge_embed = self.encoder.encode_text(self.triple_serialize(edge))
            triple_bias = np.array([self.edge_rectification(GGS, KG_edge[0], KG_edge[2]) for KG_edge in KG_edges])

            sim_scores = torch.tensor((neighbor_embeds * schema_edge_embed).sum(-1) + triple_bias)
            max_ids = sim_scores.argsort()
            id_pos = len(max_ids)-1
            idx = max_ids[id_pos]
            
            while True:
                if sim_scores[idx] < self.breadth_theta or id_pos < 0: break
                if KG_edges[idx] in reasoning_graph:
                    id_pos -= 1
                    idx = max_ids[id_pos]
                else:
                    matched_edges.append(KG_edges[idx])
                    all_scores.append(S + sim_scores[idx])
                    if len(matched_edges) >= 12: break
                    id_pos -= 1
                    idx = max_ids[id_pos]

        reasoning_graph.extend(matched_edges)
        entity_anchor_next = self.get_another(edge, entity_anchor)
        for matched_edge, S_ in zip(matched_edges, all_scores):
            similar_entity_next = self.get_another(matched_edge, similar_entity)
            self.match(entity_anchor_next, similar_entity_next, schema_graph, KG, GGS, S_, reasoning_graph, strategy, last=edge)
    
    def retrieve_isomorphic_subgraphs(
        self,
        item,
        KG: KnowledgeGraph,
        schema_graphs: KnowledgeGraph = None,
        schema_graphs_tri = None,
        strategy = 'PRECISION'
    ) -> List[KnowledgeGraph]:
    
        reasoning_graph = []
        for schema_graph, schema_graph_tri in zip(schema_graphs, schema_graphs_tri):
            GGS = self.generate_ggs_graph(KG, schema_graph_tri, item)
            for q_entity in item['q_entity']:
                entity_anchor = schema_graph.find_similiar_entities(q_entity)['best_match']
                similar_entity = KG.find_similiar_entities_with_bias(q_entity, GGS)['best_match']
                S = 0
                self.match(entity_anchor, similar_entity, schema_graph, KG, GGS, S, reasoning_graph, strategy, last=None)
        return list(set(reasoning_graph))

    def generate_ggs_graph(self, KG, schema_graph_tri, item):
        entity_ids = [self.ent2ids[x] for x in item['q_entity']]
        g_data = KGConverter(KG, item, self.ent2ids, self.rel2ids, self.ids2ent, self.ids2rel, \
                                self.entity_vectors, self.relation_vectors, entity_ids)
        num_nodes = len(KG.graph.nodes())

        question_entities_masks = (
            entities_to_mask(g_data.entity_ids_relative, num_nodes).unsqueeze(0).to('cuda')
        )
        
        triple_embeddings = []
        for triple in schema_graph_tri:
            triple_embeddings.append([self.encoder.encode_text(x) for x in triple])

        graph_retriever_input = {
            "question_embeddings": triple_embeddings, 
            "question_entities_masks": question_entities_masks,
        }
        
        ent_pred = self.graph_retriever(
            g_data.data.to('cuda'), graph_retriever_input
        )
        _, top_indices = ent_pred[0].topk(len(schema_graph_tri) * 4)
        top_entities = [self.ids2ent[y] for y in [g_data.rel_abs_map_ent[x] for x in top_indices.tolist()]]
        GGS = KG.graph.subgraph(top_entities)
        return GGS

    def entity_probs(self, KG, schema_graph_tri, item):
        entity_ids = [self.ent2ids[x] for x in item['question_entities']]
        g_data = KGConverter(KG, {'graph': item['kg_triples']}, self.ent2ids, self.rel2ids, self.ids2ent, self.ids2rel, \
                                self.entity_vectors, self.relation_vectors, entity_ids)
        num_nodes = len(KG.graph.nodes())

        question_entities_masks = (
            entities_to_mask(g_data.entity_ids_relative, num_nodes).unsqueeze(0).to('cuda')
        )
        
        triple_embedding = 0
        for triple in schema_graph_tri:
            triple_embedding += self.encoder.encode_text(self.triple_serialize(triple))
        triple_embedding /= len(schema_graph_tri)

        triple_embedding = torch.tensor(triple_embedding, device='cuda').unsqueeze(0)
        graph_retriever_input = {
            "question_embeddings": triple_embedding, 
            "question_entities_masks": question_entities_masks,
        }
        
        ent_pred = self.graph_retriever(
            g_data.data.to('cuda'), graph_retriever_input
        )
        return ent_pred, g_data.entities_list
    
    def get_stats(self) -> Dict[str, Any]:

        return {
            "knowledge_graph_stats": self.kg.get_stats() if self.kg else None,
            "vector_store_stats": self.vector_store.get_stats(),
            "cached_entity_embeddings": len(self._entity_embeddings),
            "config": self.config.model_dump()
        }