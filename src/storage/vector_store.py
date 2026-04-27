
import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    """
    
    def __init__(self, dimension: int, metric: str = "cosine"):

        self.dimension = dimension
        self.metric = metric
        self._initialized = False
    
    @abstractmethod
    def add_vectors(
        self, 
        ids: List[str], 
        vectors: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    def get_vectors(self, ids: List[str]) -> np.ndarray:
        """Get vectors by IDs."""
        pass
    
    @abstractmethod
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:

        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the store and cleanup resources."""
        pass
    
    def _compute_similarity(self, query_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:

        if self.metric == "cosine":
            # Normalize vectors
            query_norm = query_vector / np.linalg.norm(query_vector)
            candidates_norm = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
            # Compute cosine similarity
            similarities = np.dot(candidates_norm, query_norm)
        elif self.metric == "euclidean":
            # Compute negative euclidean distance
            distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
            similarities = -distances  # Negative distance as similarity
        elif self.metric == "dot":
            # Compute dot product
            similarities = np.dot(candidate_vectors, query_vector)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return similarities


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store implementation.
    """
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        super().__init__(dimension, metric)
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        logger.info(f"Initialized InMemoryVectorStore with dimension {dimension}")
    
    def add_vectors(
        self, 
        ids: List[str], 
        vectors: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the store."""
        if len(ids) != len(vectors):
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({len(vectors)})")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match store dimension {self.dimension}")
        
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError(f"Number of metadata entries ({len(metadata)}) must match number of IDs ({len(ids)})")
        
        for i, id_ in enumerate(ids):
            self.vectors[id_] = vectors[i].copy()
            if metadata is not None:
                self.metadata[id_] = metadata[i].copy()
            else:
                self.metadata[id_] = {}
        
        logger.info(f"Added {len(ids)} vectors to InMemoryVectorStore")
    
    def get_vectors(self, ids: List[str]) -> np.ndarray:
        """Get vectors by IDs."""
        vectors = []
        for id_ in ids:
            if id_ not in self.vectors:
                raise KeyError(f"Vector with ID '{id_}' not found")
            vectors.append(self.vectors[id_])
        
        return np.array(vectors)
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors.
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match store dimension {self.dimension}")
        
        # Determine candidate IDs
        if filter_ids is not None:
            candidate_ids = [id_ for id_ in filter_ids if id_ in self.vectors]
        else:
            candidate_ids = list(self.vectors.keys())
        
        if not candidate_ids:
            return [], []
        
        # Get candidate vectors
        candidate_vectors = np.array([self.vectors[id_] for id_ in candidate_ids])
        
        # Compute similarities
        similarities = self._compute_similarity(query_vector, candidate_vectors)
        
        # Get top-k results
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        result_ids = [candidate_ids[i] for i in top_indices]
        result_scores = [float(similarities[i]) for i in top_indices]
        
        return result_ids, result_scores
    
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        for id_ in ids:
            if id_ in self.vectors:
                del self.vectors[id_]
                del self.metadata[id_]
        
        logger.info(f"Deleted {len(ids)} vectors from InMemoryVectorStore")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_vectors": len(self.vectors),
            "dimension": self.dimension,
            "metric": self.metric,
            "memory_usage_mb": sum(v.nbytes for v in self.vectors.values()) / (1024 * 1024)
        }
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.vectors.clear()
        self.metadata.clear()
        logger.info("Cleared InMemoryVectorStore")
    
    def close(self) -> None:
        """Close the store."""
        self.clear()
        logger.info("Closed InMemoryVectorStore")
    
    def save_to_file(self, filepath: str) -> None:
        """Save store to file."""
        data = {
            'vectors': {k: v.tolist() for k, v in self.vectors.items()},
            'metadata': self.metadata,
            'dimension': self.dimension,
            'metric': self.metric
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved InMemoryVectorStore to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Load store from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dimension = data['dimension']
        self.metric = data['metric']
        self.vectors = {k: np.array(v) for k, v in data['vectors'].items()}
        self.metadata = data['metadata']
        
        logger.info(f"Loaded InMemoryVectorStore from {filepath}")


class RedisVectorStore(VectorStore):
    """
    Redis-based vector store implementation.
    """
    
    def __init__(
        self, 
        dimension: int, 
        metric: str = "cosine",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "stem:vector:"
    ):
        super().__init__(dimension, metric)
        
        try:
            import redis
        except ImportError:
            raise ImportError("Redis is required for RedisVectorStore. Install with: pip install redis")
        
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=False  # Keep binary data as bytes
        )
        
        self.key_prefix = key_prefix
        self.metadata_key_prefix = key_prefix.replace("vector", "metadata")
        
        # Test connection
        try:
            self.redis_client.ping()
            self._initialized = True
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def _vector_key(self, id_: str) -> str:
        """Get Redis key for vector."""
        return f"{self.key_prefix}{id_}"
    
    def _metadata_key(self, id_: str) -> str:
        """Get Redis key for metadata."""
        return f"{self.metadata_key_prefix}{id_}"
    
    def add_vectors(
        self, 
        ids: List[str], 
        vectors: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the store."""
        if len(ids) != len(vectors):
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({len(vectors)})")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match store dimension {self.dimension}")
        
        pipe = self.redis_client.pipeline()
        
        for i, id_ in enumerate(ids):
            # Store vector as pickle bytes
            vector_bytes = pickle.dumps(vectors[i])
            pipe.set(self._vector_key(id_), vector_bytes)
            
            # Store metadata as JSON
            meta = metadata[i] if metadata is not None else {}
            pipe.set(self._metadata_key(id_), json.dumps(meta))
        
        pipe.execute()
        logger.info(f"Added {len(ids)} vectors to RedisVectorStore")
    
    def get_vectors(self, ids: List[str]) -> np.ndarray:
        """Get vectors by IDs."""
        keys = [self._vector_key(id_) for id_ in ids]
        vector_data = self.redis_client.mget(keys)
        
        vectors = []
        for i, data in enumerate(vector_data):
            if data is None:
                raise KeyError(f"Vector with ID '{ids[i]}' not found")
            vectors.append(pickle.loads(data))
        
        return np.array(vectors)
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors.
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match store dimension {self.dimension}")
        
        # Get candidate IDs
        if filter_ids is not None:
            candidate_ids = filter_ids
        else:
            # Get all IDs (this might be slow for large datasets)
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            candidate_ids = [key.decode().replace(self.key_prefix, "") for key in keys]
        
        if not candidate_ids:
            return [], []
        
        # Get candidate vectors
        candidate_vectors = self.get_vectors(candidate_ids)
        
        # Compute similarities
        similarities = self._compute_similarity(query_vector, candidate_vectors)
        
        # Get top-k results
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        result_ids = [candidate_ids[i] for i in top_indices]
        result_scores = [float(similarities[i]) for i in top_indices]
        
        return result_ids, result_scores
    
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pipe = self.redis_client.pipeline()
        
        for id_ in ids:
            pipe.delete(self._vector_key(id_))
            pipe.delete(self._metadata_key(id_))
        
        pipe.execute()
        logger.info(f"Deleted {len(ids)} vectors from RedisVectorStore")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis_client.keys(pattern)
        
        return {
            "total_vectors": len(keys),
            "dimension": self.dimension,
            "metric": self.metric,
            "redis_info": self.redis_client.info()
        }
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            self.redis_client.delete(*keys)
        
        metadata_pattern = f"{self.metadata_key_prefix}*"
        metadata_keys = self.redis_client.keys(metadata_pattern)
        
        if metadata_keys:
            self.redis_client.delete(*metadata_keys)
        
        logger.info("Cleared RedisVectorStore")
    
    def close(self) -> None:
        """Close the store."""
        self.redis_client.close()
        logger.info("Closed RedisVectorStore")


def create_vector_store(
    store_type: str,
    dimension: int,
    metric: str = "cosine",
    **kwargs
) -> VectorStore:

    if store_type == "memory":
        return InMemoryVectorStore(dimension, metric)
    elif store_type == "redis":
        return RedisVectorStore(dimension, metric, **kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")