
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
from datasets import load_dataset

from core.graph import KnowledgeGraph, Entity, Relation, Triple
from core.config import Config, get_config, set_config
from core.utils import get_truth_paths, build_graph

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Dataset loader for various knowledge graph datasets."""
    
    def __init__(self):
        """Initialize dataset loader."""
        self.supported_datasets = ["freebase", "webqsp", "cwq"]
    
    def load_freebase(self, data_path: str) -> KnowledgeGraph:

        logger.info(f"Loading Freebase dataset from {data_path}")
        
        kg = KnowledgeGraph()
        data_path = Path(data_path)
        
        # Load entities
        entities_file = data_path / "entities.txt"
        if entities_file.exists():
            self._load_freebase_entities(kg, entities_file)
        
        # Load relations
        relations_file = data_path / "relations.txt"
        if relations_file.exists():
            self._load_freebase_relations(kg, relations_file)
        
        triples_file = data_path / "triples.txt"
        if triples_file.exists():
            self._load_freebase_triples(kg, triples_file)
        
        logger.info(f"Loaded Freebase KG with {len(kg.entities)} entities and {len(kg.relations)} relations")
        return kg
    
    def _load_freebase_entities(self, kg: KnowledgeGraph, entities_file: Path) -> None:
        """Load Freebase entities."""
        with open(entities_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue    
                try:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        entity_id = parts[0]
                        label = parts[1]
                        entity_type = parts[2] if len(parts) > 2 else "unknown"
                        
                        entity = Entity(
                            id=entity_id,
                            label=label,
                            entity_type=entity_type
                        )
                        kg.add_entity(entity)
                        
                except Exception as e:
                    logger.warning(f"Error parsing entity line {line_num}: {e}")
                    continue
    
    def _load_freebase_relations(self, kg: KnowledgeGraph, relations_file: Path) -> None:
        """Load Freebase relations."""
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        relation_id = parts[0]
                        subject_entity = parts[1]
                        predicate_relation = parts[2]
                        object_entity = parts[3]
                        relation_type = parts[4] if len(parts) > 4 else "unknown"
                        
                        relation = Relation(
                            id=relation_id,
                            subject_entity=subject_entity,
                            predicate_relation=predicate_relation,
                            object_entity=object_entity,
                            relation_type=relation_type
                        )
                        kg.add_relation(relation)
                        
                except Exception as e:
                    logger.warning(f"Error parsing relation line {line_num}: {e}")
                    continue
    
    def _load_freebase_triples(self, kg: KnowledgeGraph, triples_file: Path) -> None:
        """Load Freebase triples."""
        with open(triples_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        subject = parts[0]
                        predicate = parts[1]
                        object = parts[2]
                        
                        triple = Triple(
                            subject=subject,
                            predicate=predicate,
                            object=object
                        )
                        kg.add_triple(triple)
                        
                except Exception as e:
                    logger.warning(f"Error parsing triple line {line_num}: {e}")
                    continue
    
    
    def load_webqsp(self, args, data_path: str) -> KnowledgeGraph:
        dataset = load_dataset(data_path, split=args.split)

        def prepare_dataset(sample):
            graph = build_graph(sample["graph"])
            paths = get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
            ground_paths = set()
            for path in paths:
                ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
            sample["ground_paths"] = list(ground_paths)
            sample["comp_paths"] = paths
            return sample
        
        dataset = dataset.map(
            prepare_dataset,
            num_proc=24,
        )
        self.webqsp_data = dataset
    

    def save_to_pickle(self, kg: KnowledgeGraph, pickle_file: str) -> None:
        
        logger.info(f"Saving knowledge graph to pickle file: {pickle_file}")
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(kg, f)
        
        logger.info("Knowledge graph saved successfully")
    

def load_datasets(args, config: Config) -> DatasetLoader:
    """Load and prepare datasets."""
    logger = logging.getLogger(__name__)
    logger.info("Loading datasets...")
    
    loader = DatasetLoader()

    if hasattr(config, 'storage') and config.storage.webqsp_path:
        loader.load_webqsp(args, config.storage.webqsp_path)
        logger.info(f"Loaded WebQSP from {config.storage.webqsp_path}")
    
    return loader