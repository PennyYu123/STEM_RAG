
import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict
import networkx as nx
from copy import deepcopy
import json

from core.graph import KnowledgeGraph
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AllSchemaGraphs:

    def __init__(
        self,
        datasets: KnowledgeGraph = None,
    ):
        self.datasets = datasets
        self.build_schema_graphs()
    
    def build_schema_graphs(self):
        self.schema_graphs_tri_list = []

        # Temporarily using the saved schema graph triples; 
        # during actual execution, the pipeline of SGDA & SAGB models needs to be used for generation.
        with open('../schema_triples_by_SAGB.jsonl') as f: 
            self.schema_graphs_tri_list = json.load(f)

        self.schema_graphs_list = []
        for idx, item in enumerate(self.datasets):
            schema_graphs_tri, schema_graphs = self.index_triples(idx, item)
            self.schema_graphs_list.append(schema_graphs)

    def index_triples(self, idx, item: int = None) -> None:
        schema_graphs_tri = self.schema_graphs_tri_list[idx]

        schema_graphs = []
        for graph_tri in schema_graphs_tri:
            graph = KnowledgeGraph(graph_tri)
            schema_graphs.append(graph)
        return schema_graphs_tri, schema_graphs