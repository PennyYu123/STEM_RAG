
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict, Any
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
from tqdm import tqdm

from core.graph import Triple, Entity, Relation

logger = logging.getLogger(__name__)


class QwenEmbeddingEncoder:
    """Qwen3-embedding-0.6B based encoder for text and graph elements."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-embedding-0.6B",
        device: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
        cache_dir: Optional[str] = None
    ):

        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Qwen embedding encoder on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        logger.info(f"Successfully loaded Qwen embedding model: {model_name}")
    
    def encode_text(self, text: str) -> np.ndarray:

        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimension())
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling for sentence-level embeddings
            if hasattr(outputs, 'last_hidden_state'):
                # Get the last hidden states
                last_hidden_states = outputs.last_hidden_state
                # Mean pooling
                embeddings = torch.mean(last_hidden_states, dim=1)
            else:
                # Fallback to pooler output if available
                embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0]
        
        # Convert to numpy
        embedding = embeddings.cpu().numpy()[0]
        
        # Normalize if requested
        if self.normalize:
            embedding = self._normalize_vector(embedding)
        
        return embedding
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:

        if not texts:
            return np.zeros((0, self.get_embedding_dimension()))
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_states = outputs.last_hidden_state
                    batch_embeddings = torch.mean(last_hidden_states, dim=1)
                else:
                    batch_embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0]
            
            # Convert to numpy and add to results
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Normalize if requested
        if self.normalize:
            all_embeddings = self._normalize_vectors(all_embeddings)
        
        return all_embeddings
    
    def encode_entity(self, entity: Entity) -> np.ndarray:

        # Use entity label as the primary text
        text = entity
        
        return self.encode_text(text)
    
    def encode_relation(self, relation: Relation) -> np.ndarray:

        # Use relation type as the primary text
        text = relation
        
        return self.encode_text(text)
    
    def encode_triple(self, triple: Triple) -> np.ndarray:

        text = triple.to_text()
        return self.encode_text(text)
    
    def encode_triples(self, triples: List[Triple], batch_size: int = 32) -> np.ndarray:

        texts = [triple.to_text() for triple in triples]
        return self.encode_texts(texts, batch_size)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Test with a sample text to get the dimension
        sample_embedding = self.encode_text("test")
        return len(sample_embedding)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:

        if self.normalize:
            # If embeddings are normalized, dot product equals cosine similarity
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity manually
            return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def batch_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:

        if self.normalize:
            # If embeddings are normalized, use dot product
            return np.dot(candidate_embeddings, query_embedding)
        else:
            # Compute cosine similarity for each candidate
            query_norm = np.linalg.norm(query_embedding)
            candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
            dot_products = np.dot(candidate_embeddings, query_embedding)
            return dot_products / (candidate_norms * query_norm)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a single vector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize multiple vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def save_model(self, save_path: str) -> None:
        """Save the model and tokenizer to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "model"
        self.model.save_pretrained(model_path)
        
        # Save tokenizer
        tokenizer_path = save_path / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load the model and tokenizer from disk."""
        load_path = Path(load_path)
        
        # Load model
        model_path = load_path / "model"
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_path = load_path / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"Model loaded from {load_path}")


class TripleEncoder:
    """Specialized encoder for knowledge graph triples with different text templates."""
    
    def __init__(self, base_encoder: QwenEmbeddingEncoder):

        self.encoder = base_encoder
    
    def encode_triple_with_template(self, triple: Triple, template: str = "default") -> np.ndarray:

        if template == "default":
            text = triple.to_text()
        elif template == "natural":
            text = f"{triple.subject} {triple.predicate} {triple.object}"
        elif template == "structured":
            text = f"subject: {triple.subject}, predicate: {triple.predicate}, object: {triple.object}"
        elif template == "question":
            text = f"What is the {triple.predicate} of {triple.subject}? Answer: {triple.object}"
        else:
            text = triple.to_text()
        
        return self.encoder.encode_text(text)
    
    def encode_triples_with_template(self, triples: List[Triple], template: str = "default", batch_size: int = 32) -> np.ndarray:

        texts = []
        for triple in triples:
            if template == "default":
                texts.append(triple.to_text())
            elif template == "natural":
                texts.append(f"{triple.subject} {triple.predicate} {triple.object}")
            elif template == "structured":
                texts.append(f"subject: {triple.subject}, predicate: {triple.predicate}, object: {triple.object}")
            elif template == "question":
                texts.append(f"What is the {triple.predicate} of {triple.subject}? Answer: {triple.object}")
            else:
                texts.append(triple.to_text())
        
        return self.encoder.encode_texts(texts, batch_size)
    
    def compute_triple_similarity(self, triple1: Triple, triple2: Triple, template: str = "default") -> float:

        emb1 = self.encode_triple_with_template(triple1, template)
        emb2 = self.encode_triple_with_template(triple2, template)
        return self.encoder.similarity(emb1, emb2)