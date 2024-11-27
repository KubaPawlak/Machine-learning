import logging
from random import random
from typing import Callable

import numpy as np

from movie import Movie
from .decision_tree import DecisionTree as Tree
# noinspection PyProtectedMember
from .decision_tree._movie import MovieDict as MovieDict

_logger = logging.getLogger(__name__)


class _RandomFeatureSelector:
    _movie_features = {
        "movie_id",
        # "title",  # this is not a valid choice, because it uniquely defines a movie and thus is useless for generalization
        "budget",
        "genres",
        "popularity",
        "release_year",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "cast",
        "director",
    }

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.selected_features = np.random.choice(list(self._movie_features), self.num_features, replace=False)

    def transform(self, movie: Movie) -> MovieDict:
        all_features = movie.__dict__
        selected_features = {key: all_features[key] for key in self.selected_features}
        return selected_features


class RandomForestClassifier:
    def __init__(self, num_trees: int, num_features: int,
                 aggregation_function: Callable[[list[int]], float] = np.average,
                 **tree_args):
        self.fitted = False
        self.trees: list[Tree] = []
        self.num_trees = num_trees
        self.num_features = num_features
        self.tree_args = tree_args
        self.aggregation_function = aggregation_function
        pass

    def _select_random_features(self, movies: list[Movie]) -> list[MovieDict]:
        random_feature_selector = _RandomFeatureSelector(self.num_features)
        return [random_feature_selector.transform(movie) for movie in movies]

    type Bootstrap = (list[MovieDict], list[int])

    def _generate_bootstraps(self, movies: list[Movie], ratings: list[int]) -> list[Bootstrap]:
        num_movies = len(movies)
        # combine movies with ratings, so they do not get mixed up during random selection
        movies_with_ratings: list[(Movie, int)] = list(zip(movies, ratings))

        bootstraps = []
        random_generator: np.random.Generator = np.random.default_rng()
        for _ in range(self.num_trees):
            selected_movies = random_generator.choice(movies_with_ratings, num_movies, replace=True, axis=0)
            movies: list[Movie] = selected_movies[:,0].tolist()
            ratings: list[int] = selected_movies[:,1].tolist() # convert list of tuples into tuple of lists
            movies: list[MovieDict] = self._select_random_features(movies)
            bootstrap = (movies, ratings)
            bootstraps.append(bootstrap)

        return bootstraps

    def _fit_tree(self, bootstrap: Bootstrap) -> Tree:
        movies, ratings = bootstrap
        tree = Tree(*self.tree_args)
        tree.fit(movies, ratings)
        return tree

    def fit(self, movies: list[Movie], ratings: list[int]):
        bootstraps = self._generate_bootstraps(movies, ratings)
        self.trees = list(map(self._fit_tree, bootstraps))
        pass

    def _aggregate_predictions(self, predictions: list[int]) -> int:
        result = int(round(self.aggregation_function(predictions), 0))
        assert 0 <= result <= 5
        return result

    def _predict_single(self, movie: Movie) -> int:
        predictions: list[int] = list(map(lambda tree: tree.predict([movie]), self.trees))
        return self._aggregate_predictions(predictions)

    def predict(self, movies: list[Movie]) -> list[int]:
        return list(map(self._predict_single, movies))
