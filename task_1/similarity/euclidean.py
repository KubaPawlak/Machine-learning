import numpy as np

def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    euclidean_distance = np.sqrt(np.sum(np.square(a - b)))
    return 1 / (1 + euclidean_distance)