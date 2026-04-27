
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_embeddings(embeddings: np.ndarray, axis: int = 1) -> np.ndarray:
   
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:

    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def compute_batch_similarity(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:

    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return np.zeros(len(candidate_embeddings))
    
    query_normalized = query_embedding / query_norm
    
    # Normalize candidate embeddings
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    candidate_norms[candidate_norms == 0] = 1  # Avoid division by zero
    candidates_normalized = candidate_embeddings / candidate_norms[:, np.newaxis]
    
    # Compute similarities
    similarities = np.dot(candidates_normalized, query_normalized)
    
    return similarities


def compute_euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:

    return float(np.linalg.norm(embedding1 - embedding2))


def compute_manhattan_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:

    return float(np.sum(np.abs(embedding1 - embedding2)))


def compute_mahalanobis_distance(
    embedding1: np.ndarray, 
    embedding2: np.ndarray, 
    covariance_matrix: np.ndarray
) -> float:

    diff = embedding1 - embedding2
    
    # Ensure covariance matrix is invertible
    try:
        inv_covariance = np.linalg.inv(covariance_matrix)
        distance = np.sqrt(np.dot(np.dot(diff.T, inv_covariance), diff))
        return float(distance)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix is singular, falling back to Euclidean distance")
        return compute_euclidean_distance(embedding1, embedding2)


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:

    if len(embeddings) == 0:
        return {}
    
    stats = {
        "mean": np.mean(embeddings, axis=0),
        "std": np.std(embeddings, axis=0),
        "min": np.min(embeddings, axis=0),
        "max": np.max(embeddings, axis=0),
        "median": np.median(embeddings, axis=0),
        "dimension": embeddings.shape[1],
        "num_embeddings": embeddings.shape[0]
    }
    
    # Compute pairwise similarities
    if embeddings.shape[0] > 1:
        # Sample a subset for large datasets
        sample_size = min(1000, embeddings.shape[0])
        indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(sample_embeddings)):
            for j in range(i + 1, len(sample_embeddings)):
                similarity = compute_cosine_similarity(sample_embeddings[i], sample_embeddings[j])
                similarities.append(similarity)
        
        stats["pairwise_similarities"] = {
            "mean": np.mean(similarities),
            "std": np.std(similarities),
            "min": np.min(similarities),
            "max": np.max(similarities)
        }
    
    return stats


def reduce_dimensionality_pca(embeddings: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:

    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    return reduced_embeddings, pca.explained_variance_ratio_


def reduce_dimensionality_tsne(
    embeddings: np.ndarray, 
    n_components: int = 2, 
    perplexity: int = 30,
    random_state: int = 42
) -> np.ndarray:

    from sklearn.manifold import TSNE
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000
    )
    
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    return reduced_embeddings


def cluster_embeddings(
    embeddings: np.ndarray, 
    n_clusters: int,
    method: str = "kmeans",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:

    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
    elif method == "dbscan":
        clusterer = DBSCAN(**kwargs)
    elif method == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Get cluster centers for methods that support it
    if hasattr(clusterer, 'cluster_centers_'):
        cluster_centers = clusterer.cluster_centers_
    else:
        # Compute centers manually
        cluster_centers = []
        for cluster_id in np.unique(cluster_labels):
            cluster_embeddings = embeddings[cluster_labels == cluster_id]
            cluster_center = np.mean(cluster_embeddings, axis=0)
            cluster_centers.append(cluster_center)
        cluster_centers = np.array(cluster_centers)
    
    return cluster_labels, cluster_centers


def find_outliers(embeddings: np.ndarray, method: str = "isolation_forest", **kwargs) -> np.ndarray:

    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    
    if method == "isolation_forest":
        detector = IsolationForest(random_state=42, **kwargs)
    elif method == "lof":
        detector = LocalOutlierFactor(**kwargs)
    elif method == "one_class_svm":
        detector = OneClassSVM(**kwargs)
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    if method == "lof":
        outlier_labels = detector.fit_predict(embeddings)
        outliers = outlier_labels == -1
    else:
        detector.fit(embeddings)
        outlier_labels = detector.predict(embeddings)
        outliers = outlier_labels == -1
    
    return outliers


def compute_similarity_matrix(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:

    n_samples = embeddings.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i, n_samples):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                if metric == "cosine":
                    similarity = compute_cosine_similarity(embeddings[i], embeddings[j])
                elif metric == "euclidean":
                    similarity = -compute_euclidean_distance(embeddings[i], embeddings[j])
                elif metric == "dot":
                    similarity = float(np.dot(embeddings[i], embeddings[j]))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
    
    return similarity_matrix


def select_representative_embeddings(
    embeddings: np.ndarray, 
    n_representatives: int,
    method: str = "kmeans"
) -> Tuple[np.ndarray, np.ndarray]:

    if method == "kmeans":
        # Use k-means clustering and select cluster centers
        cluster_labels, cluster_centers = cluster_embeddings(embeddings, n_representatives, method="kmeans")
        return cluster_centers, np.arange(len(cluster_centers))
    
    elif method == "random":
        # Random selection
        indices = np.random.choice(len(embeddings), n_representatives, replace=False)
        return embeddings[indices], indices
    
    elif method == "farthest":
        # Farthest point sampling
        n_samples = embeddings.shape[0]
        selected_indices = []
        
        # Start with a random point
        selected_indices.append(np.random.randint(n_samples))
        
        # Iteratively select farthest points
        for _ in range(1, n_representatives):
            # Compute distances to already selected points
            remaining_indices = [i for i in range(n_samples) if i not in selected_indices]
            
            if not remaining_indices:
                break
            
            # Find the point farthest from already selected points
            max_min_distance = -1
            farthest_idx = remaining_indices[0]
            
            for idx in remaining_indices:
                # Compute minimum distance to already selected points
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = compute_euclidean_distance(embeddings[idx], embeddings[selected_idx])
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    farthest_idx = idx
            
            selected_indices.append(farthest_idx)
        
        selected_indices = np.array(selected_indices)
        return embeddings[selected_indices], selected_indices
    
    else:
        raise ValueError(f"Unsupported selection method: {method}")


def validate_embeddings(embeddings: np.ndarray) -> Dict[str, Any]:

    validation_results = {
        "is_valid": True,
        "issues": [],
        "warnings": []
    }
    
    if np.isnan(embeddings).any():
        validation_results["is_valid"] = False
        validation_results["issues"].append("Embeddings contain NaN values")
    
    if np.isinf(embeddings).any():
        validation_results["is_valid"] = False
        validation_results["issues"].append("Embeddings contain infinite values")
    
    zero_vectors = np.all(embeddings == 0, axis=1)
    if zero_vectors.any():
        validation_results["warnings"].append(f"Found {zero_vectors.sum()} zero vectors")
    
    max_abs_value = np.max(np.abs(embeddings))
    if max_abs_value > 1e6:
        validation_results["warnings"].append(f"Found very large values (max: {max_abs_value})")
    
    min_abs_value = np.min(np.abs(embeddings[embeddings != 0]))
    if min_abs_value < 1e-10:
        validation_results["warnings"].append(f"Found very small non-zero values (min: {min_abs_value})")
    
    if len(embeddings.shape) != 2:
        validation_results["is_valid"] = False
        validation_results["issues"].append(f"Embeddings should be 2D array, got {len(embeddings.shape)}D")
    
    return validation_results