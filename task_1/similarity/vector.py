import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    euclidean_distance = np.sqrt(np.sum(np.square(a - b)))
    return 1 / (1 + euclidean_distance)


def manhattan_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Manhattan similarity between two vectors."""
    manhattan_distance = np.sum(np.abs(vec_a - vec_b))
    return 1 / (1 + manhattan_distance)
