import networkx as nx
import random
import matplotlib.pyplot as plt
import json
import os
import argparse
import tqdm
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

blacklist = ['freebase.valuenotation.is_reviewed', 'common.topic.notable_types', 'common.topic.notable_for']

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
            neighbors.append({
                "direction": "out",
                "relation": attr['relation'],
                "neighbor": target
            })

        for source, _, attr in self.graph.in_edges(entity, data=True):
            neighbors.append({
                "direction": "in", 
                "relation": attr['relation'],
                "neighbor": source
            })
            
        return neighbors

    def add_next_hop(self, next_node, next_edge, visited_nodes, sub_graph):

        visited_nodes.add(next_node)
        sub_graph.append(next_edge)
        return visited_nodes, sub_graph

    
    def random_walk_subgraph(self, start_node=None, min_nodes=3, max_nodes=6):

        if not self.nodes_list:
            return None
            
        while True:
            start_node = random.choice(self.nodes_list)
            if 'm.' in start_node or 'g.' in start_node: 
                continue
            break
            
        visited_nodes = {start_node}
        current_node = start_node
        target_size = random.randint(min_nodes, max_nodes)
        sub_graph = []

        max_attempts = 10 
        attempts = 0
        branch = 1
        
        while len(visited_nodes) < target_size and attempts < max_attempts:
            if attempts >= 9: pass
            attempts += 1

            candidates = []
            for _, target, attr in self.graph.out_edges(current_node, data=True):
                candidates.append({
                    "direction": "out",
                    "relation": attr['relation'],
                    "neighbor": target, 
                    "now": current_node
                })

            for source, _, attr in self.graph.in_edges(current_node, data=True):
                candidates.append({
                    "direction": "in",
                    "relation": attr['relation'],
                    "neighbor": source,
                    "now": current_node
                })

            candidates = [x for x in candidates if not (x['neighbor'] in visited_nodes or x['relation'] in blacklist or 'm.' in x['neighbor'] or 'g.' in x['neighbor'])]

            if not candidates:
                current_node = random.choice(list(visited_nodes))
                continue

            if branch and len(candidates) >=2 and random.random() < 0.1:
                remain_size = target_size - len(visited_nodes)
                size_1 = random.randint(1, remain_size - 1) if remain_size > 2 else 1
                size_2 = remain_size - size_1 if remain_size - size_1 > 0 else 1

                next_edges = random.sample(candidates, 2)
                next_node = [x['neighbor'] for x in next_edges]

                current_node = next_node[0]
                current_edge = next_edges[0]
                for _ in range(size_1):
                    visited_nodes, sub_graph = self.add_next_hop(current_node, current_edge, visited_nodes, sub_graph)

                    candidates = []
                    for _, target, attr in self.graph.out_edges(current_node, data=True):
                        candidates.append({
                            "direction": "out",
                            "relation": attr['relation'],
                            "neighbor": target, 
                            "now": current_node
                        })

                    for source, _, attr in self.graph.in_edges(current_node, data=True):
                        candidates.append({
                            "direction": "in",
                            "relation": attr['relation'],
                            "neighbor": source,
                            "now": current_node
                        })

                    candidates = [x for x in candidates if not (x['neighbor'] in visited_nodes or x['relation'] in blacklist or 'm.' in x['neighbor'] or 'g.' in x['neighbor'])]

                    if not candidates:
                        break
                    next_edge = random.choice(candidates)
                    next_node = next_edge['neighbor']

                    current_node = next_node
                    current_edge = next_edge
                visited_nodes, sub_graph = self.add_next_hop(current_node, current_edge, visited_nodes, sub_graph)

                current_node = next_node[1]
                current_edge = next_edges[1]
                for _ in range(size_2):
                    visited_nodes, sub_graph = self.add_next_hop(current_node, current_edge, visited_nodes, sub_graph)

                    candidates = []
                    for _, target, attr in self.graph.out_edges(current_node, data=True):
                        candidates.append({
                            "direction": "out",
                            "relation": attr['relation'],
                            "neighbor": target, 
                            "now": current_node
                        })

                    for source, _, attr in self.graph.in_edges(current_node, data=True):
                        candidates.append({
                            "direction": "in",
                            "relation": attr['relation'],
                            "neighbor": source,
                            "now": current_node
                        })

                    candidates = [x for x in candidates if not (x['neighbor'] in visited_nodes or x['relation'] in blacklist or 'm.' in x['neighbor'])]

                    if not candidates:
                        break
                    next_edge = random.choice(candidates)
                    next_node = next_edge['neighbor']

                    current_node = next_node
                    current_edge = next_edge
                visited_nodes, sub_graph = self.add_next_hop(current_node, current_edge, visited_nodes, sub_graph)
            else:
                next_edge = random.choice(candidates)
                next_node = next_edge['neighbor']
                visited_nodes, sub_graph = self.add_next_hop(next_node, next_edge, visited_nodes, sub_graph)
                current_node = next_node
        
        return sub_graph

def process(sample, undirected=False):
    
    start_nodes = sample['q_entity']
    answer_nodes = sample['a_entity']
    kg = KnowledgeGraph(sample['graph'])

    sub_g = []
    try_num = 0
    while len(sub_g) < 1 and try_num < 20:
        try:
            sub_g = kg.random_walk_subgraph(start_node=None, min_nodes=2, max_nodes=5)
        except:
            continue
        try_num += 1

    if not sub_g: return None
    triples = []
    for edge in sub_g:
        if edge['direction'] == 'in':
            triples.append((edge['neighbor'], edge['relation'], edge['now']))
        else:
            triples.append((edge['now'], edge['relation'], edge['neighbor']))

    triples = list(set(triples))
    triples_n = deepcopy(triples)
    question_e = []

    all_entities = []
    for triple in triples_n:
        if not triple[0] in all_entities: all_entities.append(triple[0])
        if not triple[2] in all_entities: all_entities.append(triple[2])
    qe_n = random.randint(1, len(all_entities) - 1)
    mask_n = len(all_entities) - qe_n
    question_e = random.sample(all_entities, qe_n)

    all_masked = []
    for ent in all_entities:
        if ent in question_e: continue
        all_masked.append(ent)
    random.shuffle(all_masked)
    mask_plh_map = {ent: f"[ENT{i+1}]" for i, ent in enumerate(all_masked)}

    for idx in range(len(triples_n)):
        if triples_n[idx][0] in mask_plh_map or triples_n[idx][2] in mask_plh_map:
            new_triple = list(triples_n[idx])
            if triples_n[idx][0] in mask_plh_map:
                new_triple[0] = mask_plh_map[triples_n[idx][0]]
            if triples_n[idx][2] in mask_plh_map:
                new_triple[2] = mask_plh_map[triples_n[idx][2]]
            new_triple = tuple(new_triple)
            triples_n[idx] = new_triple

    return {'triples': triples_n, 'answer': all_masked[-1], 'raw': triples, 'question_entities': question_e}
    

def index_graph(args):
    if args.d == 'webqsp':
        input_file = '../STEM-RAG/data/webqsp'
        dataset = load_dataset(input_file, split=args.split)
    if args.d == 'cwq':
        input_file = '../STEM-RAG/data/cwq'
        dataset = load_dataset(input_file, split=args.split)

    results = []
    t = tqdm.tqdm()
    for item in dataset:
        for _ in range(10):
            item_triple = process(item)
            if item_triple is not None:
                results.append(item_triple)
                t.update(1)
        if len(results) >= 500000:
            break

    json.dump(results, open(args.output_path, 'w'), indent=4)
    index_dataset = Dataset.from_list(results)
    index_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--d', '-d', type=str, default='webqsp')
    argparser.add_argument('--split', type=str, default='train')
    argparser.add_argument('--output_path', type=str, default='../masked_sample_subgraphs.jsonl')
    argparser.add_argument('--n', type=int, default=36, help='number of processes')
    
    args = argparser.parse_args()
    index_graph(args)
