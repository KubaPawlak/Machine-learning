import logging
from data.movie import train as global_train
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from task_2.classification.decision_tree import DecisionTree

NUM_RUNS = 3

def test_parameters(max_depth: int,
                    min_samples_split: int) -> float:
    """Calculate the accuracy of the Decision Tree model for given parameters."""

    scores = []
    for i in range(NUM_RUNS):
        logging.info("Running iteration %i", i + 1)
        train, test = train_test_split(global_train, test_size=0.1, train_size=0.4)
        train: pd.DataFrame
        test: pd.DataFrame

        # Prepare training and test data
        X_train = train.drop(["Rating"], axis=1).values
        y_train = train["Rating"].values
        X_test = test.drop(["Rating"], axis=1).values
        y_test = test["Rating"].values

        # Initialize and train the Decision Tree
        tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        tree.fit(X_train, y_train)

        # Make predictions
        predictions = tree.predict(X_test)

        # Calculate accuracy
        scores.append(accuracy_score(y_true=y_test, y_pred=predictions))

    return float(np.average(np.array(scores)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parameters = {
        'max_depth': 10,
        'min_samples_split': 2
    }

    accuracy = test_parameters(
        max_depth=parameters['max_depth'],
        min_samples_split=parameters['min_samples_split'],
    )

    print("=== Parameters: ===")
    for key, value in parameters.items():
        print(f"\t{key}: {value}")
    print()
    print("Accuracy: ", accuracy)