import logging
from collections.abc import Callable
from pathlib import Path

from task_1.classification import KNeighborsClassifier
from task_1.movie import Movie
from task_1.movie.tmdb.client import Client
from task_1.similarity import calculate_movie_similarity, fit_scaler

_RESULT_FILE = Path('task_1_result.csv').absolute()
import pandas as pd

_tmdb_client = Client()


def generate_predictions(train: pd.DataFrame,
                         task: pd.DataFrame,
                         n_neighbors: int = 5,
                         similarity_fn: Callable[[Movie, Movie], float] = calculate_movie_similarity
                         ) -> pd.DataFrame:
    logging.debug("Fetching movie data")
    all_movies = [_tmdb_client.get_movie(movie_id) for movie_id in train['MovieID'].unique()]
    logging.debug("Fitting scaler")
    fit_scaler(all_movies)

    def predict_rating(row: pd.Series) -> int:
        user_id, movie_id = row['UserID'], row['MovieID']
        logging.debug("Predicting rating for movie %i by user %i", movie_id, user_id)

        def distance_function(movie_1: Movie, movie_2: Movie) -> float:
            # invert the result to convert from similarity (decreasing) to distance (increasing)
            return -1 * similarity_fn(movie_1, movie_2)

        classifier = KNeighborsClassifier(n_neighbors, distance_function)
        movie_to_predict = _tmdb_client.get_movie(movie_id)

        watched_movies_ratings: pd.DataFrame = train[train['UserID'] == user_id]

        # get list of movies watched by the user
        watched_movies: list[Movie] = [movie for movie in all_movies if
                                       movie.movie_id in watched_movies_ratings['MovieID'].values]

        logging.debug("Generating prediction")
        predicted_rating = classifier.fit_predict(watched_movies, watched_movies_ratings['Rating'].to_numpy(),
                                                  movie_to_predict)
        logging.debug("Predicted rating of movie %i by user %i: %i", movie_id, user_id, predicted_rating)
        return predicted_rating

    # end predict_rating

    logging.info("Calculating predictions...")
    # apply the prediction function to each row in the task dataframe
    predicted = task.apply(predict_rating, axis=1).astype(int)

    task_with_predictions = task.copy()
    task_with_predictions['Rating'] = predicted
    return task_with_predictions


def _main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    from data.movie import task, train
    task_with_predictions = generate_predictions(train, task)

    task_with_predictions.to_csv(_RESULT_FILE, index=False, sep=';')
    logging.info("Written results file to %s", _RESULT_FILE)


if __name__ == '__main__':
    _main()
