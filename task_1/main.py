import logging
from pathlib import Path

from data.movie import task, train
from task_1.classifier import KNeighborsClassifier
from task_1.movie import Movie
from task_1.movie.tmdb.client import Client
from features.movie_similarity import calculate_movie_similarity

_RESULT_FILE = Path('task_1_result.csv').absolute()
import pandas as pd

tmdb_client = Client()


def predict_score(user_id: int, movie_id: int) -> int:

    logging.debug("Predicting rating for movie %i by user %i", movie_id, user_id)

    def distance_function(movie_1: Movie, movie_2: Movie) -> float:
        return 1 / calculate_movie_similarity(movie_1, movie_2)

    classifier = KNeighborsClassifier(n_neighbors=5, distance_function=distance_function)
    movie_to_predict = tmdb_client.get_movie(movie_id)

    watched_movies_ratings = train[train['UserID'] == user_id]

    # get list of features for the movies watched by the user
    watched_movies: pd.Series[int] = watched_movies_ratings['MovieID']
    assert movie_id not in watched_movies.values
    logging.debug("Gathering movies watched by user %i...", user_id)
    watched_movies: list[Movie] = [tmdb_client.get_movie(movie_id) for movie_id in watched_movies]

    logging.debug("Generating prediction")
    predicted_rating =  classifier.fit_predict(watched_movies, watched_movies_ratings['Rating'].to_numpy(), movie_to_predict)
    logging.info("Predicted rating of movie %i by user %i: %i", movie_id, user_id, predicted_rating)
    return predicted_rating


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    logging.info("Calculating predictions...")
    # apply the prediction function to each row in the task dataframe
    predicted = task.apply(lambda row: predict_score(row['UserID'], row['MovieID']), axis=1).astype(int)

    task_with_predictions = task.copy()
    task_with_predictions['Rating'] = predicted

    task_with_predictions.to_csv(_RESULT_FILE, index=False, sep=';')
    logging.info("Written results file to %s", _RESULT_FILE)


if __name__ == '__main__':
    main()
