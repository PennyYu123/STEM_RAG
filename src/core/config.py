
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import yaml


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    model_name: str = Field(default="qwen3-embedding-0.6B", description="Embedding model name")
    model_path: Optional[str] = Field(default=None, description="Local model path")
    device: str = Field(default="auto", description="Device to run model on")
    batch_size: int = Field(default=32, description="Batch size for encoding")
    max_length: int = Field(default=512, description="Maximum sequence length")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")


class StorageConfig(BaseModel):
    """Configuration for vector storage."""
    backend: str = Field(default="redis", description="Storage backend (redis, memory, disk)")
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    disk_path: str = Field(default="./data/vectors", description="Path for disk storage")
    cache_size: int = Field(default=1000000, description="Maximum cache size")
    webqsp_path: str = Field(default="None", description="Path to WebQSP dataset")
    cwq_path: str = Field(default="None", description="Path to CWQ dataset")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval algorithms."""
    top_k: int = Field(default=10, description="Number of top results to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    max_candidates: int = Field(default=1000, description="Maximum candidates for filtered search")
    use_gpu: bool = Field(default=False, description="Whether to use GPU for similarity computation")


class GraphConfig(BaseModel):
    """Configuration for graph processing."""
    max_depth: int = Field(default=3, description="Maximum search depth")
    max_paths: int = Field(default=100, description="Maximum number of paths to consider")
    edge_weight_threshold: float = Field(default=0.5, description="Minimum edge weight")
    entity_placeholder: str = Field(default="[ENTX]", description="Entity placeholder token")

class GNNConfig(BaseModel):
    """Configuration for T-GNN."""
    input_dim: int = Field(default=512)
    rel_emb_dim: int = Field(default=512)
    hidden_dims: list = Field(default=[512, 512, 512, 512, 512, 512])
    message_func: str = Field(default='distmult')
    aggregate_func: str = Field(default='sum')

class APIConfig(BaseModel):
    """Configuration for API server."""
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=1, description="Number of worker processes")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Maximum request size in bytes")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_bytes: int = Field(default=10 * 1024 * 1024, description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of backup log files")


class Config(BaseModel):
    """Main configuration class."""
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Embedding config
        embedding_config = {}
        if os.getenv("EMBEDDING_MODEL_NAME"):
            embedding_config["model_name"] = os.getenv("EMBEDDING_MODEL_NAME")
        if os.getenv("EMBEDDING_DEVICE"):
            embedding_config["device"] = os.getenv("EMBEDDING_DEVICE")
        if os.getenv("EMBEDDING_BATCH_SIZE"):
            embedding_config["batch_size"] = int(os.getenv("EMBEDDING_BATCH_SIZE"))
        
        if embedding_config:
            config_dict["embedding"] = embedding_config
        
        # Storage config
        storage_config = {}
        if os.getenv("STORAGE_BACKEND"):
            storage_config["backend"] = os.getenv("STORAGE_BACKEND")
        if os.getenv("REDIS_HOST"):
            storage_config["redis_host"] = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            storage_config["redis_port"] = int(os.getenv("REDIS_PORT"))
        
        if storage_config:
            config_dict["storage"] = storage_config
        
        # Retrieval config
        retrieval_config = {}
        if os.getenv("RETRIEVAL_TOP_K"):
            retrieval_config["top_k"] = int(os.getenv("RETRIEVAL_TOP_K"))
        if os.getenv("SIMILARITY_THRESHOLD"):
            retrieval_config["similarity_threshold"] = float(os.getenv("SIMILARITY_THRESHOLD"))
        
        if retrieval_config:
            config_dict["retrieval"] = retrieval_config
        
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)
    
    def get_embedding_model_path(self) -> str:
        """Get the full path to the embedding model."""
        if self.embedding.model_path:
            return self.embedding.model_path
        
        # Default model path
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        return str(model_dir / self.embedding.model_name)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config