from collections.abc import Callable

import numpy as np

from task_1.movie import Movie


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int, distance_function: Callable[[Movie, Movie], float]) -> None:
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')

        self.n_neighbors = n_neighbors
        self.distance = distance_function

    def fit_predict(self, watched_movies: list[Movie], labels: np.ndarray, movie_to_predict: Movie) -> int:
        """
        :param watched_movies: past watched movies to evaluate against.
        :param labels: labels of the data points to evaluate against.
        :param movie_to_predict: similarity of the point to be predicted.
        :return: predicted label.
        """
        assert len(watched_movies) == len(labels), "Movies and labels arrays have different sizes"
        distance = lambda x: self.distance(x, movie_to_predict)

        # calculate distance to each data point
        point_distances = np.array(list(map(distance, watched_movies)))

        # get k indices with the smallest distance
        k_smallest_indices = np.argpartition(point_distances, self.n_neighbors - 1)[:self.n_neighbors]

        # retrieve the labels for the k closest points
        k_closest_labels = labels[k_smallest_indices]

        # return the label that is most common
        # using numpy bincount, because it is known that the labels will be integers from the range 1-5
        counts = np.bincount(k_closest_labels)
        return np.argmax(counts).item()
