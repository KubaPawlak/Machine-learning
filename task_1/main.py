import logging
import time
from collections.abc import Callable
from pathlib import Path

import tqdm
from numpy import ndarray

import util.validation
from data.movie import train
from movie import Movie
from movie.tmdb.client import Client
from task_1.classification import KNeighborsClassifier
from task_1.similarity import calculate_movie_similarity, fit_scaler
from util import submission

_RESULT_FILE = Path('task_1_result.csv').absolute()
import pandas as pd



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_classifier() -> KNeighborsClassifier:
    def distance(movie_1: Movie, movie_2: Movie) -> float:
        similarity = calculate_movie_similarity(movie_1, movie_2,
                                          metric='euclidean',
                                          scalar_similarity_part=0.6,
                                          genres_similarity_part=0.1,
                                          cast_similarity_part=0.1,
                                          directors_similarity_part=0.1,
                                          ratings_similarity_part=0.1
                                          )
        return 1 / similarity

    return KNeighborsClassifier(n_neighbors=5, distance_function=distance)


class KNNModel(submission.Model[tuple[list[Movie], ndarray]]):
    def __init__(self, training_data: pd.DataFrame):
        super().__init__(training_data, per_user=True)
        tmdb_client = Client()
        self.all_movies: dict[int, Movie] = {movie_id: tmdb_client.get_movie(movie_id)
                                             for movie_id in training_data['MovieID'].unique()}
        fit_scaler(self.all_movies.values())

    def create_model(self, training_data: pd.DataFrame) -> tuple[list[Movie], ndarray]:
        assert training_data['UserID'].nunique() == 1
        user_id = training_data['UserID'].unique()[0]

        watched_movies_ratings = training_data[training_data['UserID'] == user_id]['Rating'].values
        watched_movies: list[Movie] = [self.all_movies[id_] for id_ in training_data['MovieID'].values]

        return watched_movies, watched_movies_ratings

    def predict(self, model: tuple[list[Movie], ndarray], user_ids: list[int], movie_ids: list[int]) -> list[int]:
        movies, ratings = model
        assert pd.Series(user_ids).nunique() == 1

        classifier = create_classifier()

        results = []
        for movie_id in movie_ids:
            movie_to_predict = self.all_movies[movie_id]
            result = classifier.fit_predict(movies, ratings, movie_to_predict)
            results.append(result)
        return results



def _main() -> None:
    logging.basicConfig(level=logging.INFO)

    model = KNNModel(train)
    validator = util.validation.Validator(model)
    validator.train_set_accuracy()



if __name__ == '__main__':
    _main()
