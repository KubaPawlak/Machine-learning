import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data.movie import train as global_train
from task_1.main import generate_predictions
from task_1.similarity.movie_similarity import calculate_movie_similarity, create_rating_matrix

NUM_RUNS = 1


def test_parameters(n_neighbors: int) -> float:
    """Calculate the accuracy of the model for given parameters."""

    scores = []
    for i in range(NUM_RUNS):
        logging.info("Running iteration %i", i + 1)
        train, test = train_test_split(global_train, test_size=0.1, train_size=0.5)
        train: pd.DataFrame
        test: pd.DataFrame

        task = test.copy()
        task['Rating'] = np.nan

        rating_matrix = create_rating_matrix(train)

        # override similarity function by using only selected subset of training data
        def similarity_fn(movie_1, movie_2) -> float:
            return calculate_movie_similarity(movie_1, movie_2, rating_matrix=rating_matrix)

        predictions = generate_predictions(train, task, n_neighbors=n_neighbors, similarity_fn=similarity_fn)

        scores.append(accuracy_score(y_true=test['Rating'], y_pred=predictions['Rating']))

    return float(np.average(np.array(scores)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parameters = {
        'n_neighbors': 3
    }

    accuracy = test_parameters(
        n_neighbors=parameters['n_neighbors']
    )

    print("=== Parameters: ===")
    for key, value in parameters.items():
        print(f"\t{key}: {value}")
    print()
    print("Accuracy: ", accuracy)
