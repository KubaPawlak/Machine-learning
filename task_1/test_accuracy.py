import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data.movie import train as global_train
from task_1.main import generate_predictions
from task_1.movie.tmdb.client import Client
from task_1.similarity import fit_scaler
from task_1.similarity.movie_similarity import calculate_movie_similarity, create_rating_matrix

NUM_RUNS = 3


def test_parameters(n_neighbors: int,
                    metric: str,
                    scalar_similarity_part: float = (6.0 / 10),
                    genres_similarity_part: float = (1.0 / 10),
                    cast_similarity_part: float = (1.0 / 10),
                    directors_similarity_part: float = (1.0 / 10),
                    ratings_similarity_part: float = (1.0 / 10)
                    ) -> float:
    """Calculate the accuracy of the model for given parameters."""

    scores = []
    for i in range(NUM_RUNS):
        logging.info("Running iteration %i", i + 1)
        train, test = train_test_split(global_train, test_size=0.1, train_size=0.4)
        train: pd.DataFrame
        test: pd.DataFrame

        task = test.copy()
        task['Rating'] = np.nan

        rating_matrix = create_rating_matrix(train)

        # override similarity function by using only selected subset of training data
        def similarity_fn(movie_1, movie_2) -> float:
            return calculate_movie_similarity(movie_1, movie_2,
                                              rating_matrix=rating_matrix,
                                              metric=metric,
                                              scalar_similarity_part=scalar_similarity_part,
                                              genres_similarity_part=genres_similarity_part,
                                              cast_similarity_part=cast_similarity_part,
                                              directors_similarity_part=directors_similarity_part,
                                              ratings_similarity_part=ratings_similarity_part
                                              )

        predictions = generate_predictions(train, task, n_neighbors=n_neighbors, similarity_fn=similarity_fn)

        scores.append(accuracy_score(y_true=test['Rating'], y_pred=predictions['Rating']))

    return float(np.average(np.array(scores)))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parameters = {
        'n_neighbors': 17,
        'metric': 'euclidean',
        'scalar_similarity_part': 6,
        'genres_similarity_part': 1,
        'cast_similarity_part': 1,
        'directors_similarity_part': 1,
        'ratings_similarity_part': 1,
    }

    accuracy = test_parameters(
        n_neighbors=parameters['n_neighbors'],
        metric=parameters['metric'],
        scalar_similarity_part=parameters['scalar_similarity_part'],
        genres_similarity_part=parameters['genres_similarity_part'],
        cast_similarity_part=parameters['cast_similarity_part'],
        directors_similarity_part=parameters['directors_similarity_part'],
        ratings_similarity_part=parameters['ratings_similarity_part'],
    )

    print("=== Parameters: ===")
    for key, value in parameters.items():
        print(f"\t{key}: {value}")
    print()
    print("Accuracy: ", accuracy)
