from collections.abc import Callable

import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int, similarity_function: Callable[[np.ndarray, np.ndarray], float]) -> None:
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')

        self.n_neighbors = n_neighbors
        self.similarity = similarity_function

    def fit_predict(self, features: np.ndarray, labels: np.ndarray, prediction_features: np.ndarray) -> int:
        """
        :param features: features of the data points to evaluate against.
        :param labels: labels of the data points to evaluate against.
        :param prediction_features: features of the point to be predicted.
        :return: predicted label.
        """
        similarity = lambda x: self.similarity(x, prediction_features)

        # calculate distance to each data point
        point_similarities = np.apply_along_axis(similarity, 1, features)
        # get k indices with the smallest distance
        k_smallest_indices = np.argpartition(point_similarities, self.n_neighbors - 1)[:self.n_neighbors]

        # retrieve the labels for the k closest points
        k_closest_labels = labels[k_smallest_indices]

        # return the label that is most common
        # using numpy bincount, because it is known that the labels will be integers from the range 1-5
        counts = np.bincount(k_closest_labels)
        return np.argmax(counts).item()
