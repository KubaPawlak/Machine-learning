import numpy as np


def manhattan_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Manhattan distance between two vectors."""
    return np.sum(np.abs(vec_a - vec_b))
