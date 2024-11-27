import logging

from data.movie import train, task
from movie import Movie
from task_2.classification.decision_tree import DecisionTree

logging.basicConfig(level=logging.INFO)


def main():
    # Preprocess data
    logging.info("Preprocessing data...")
    try:
        train.dropna(inplace=True)
        X_train = train.drop(["Rating"], axis=1).values
        y_train = train["Rating"].values

        task.fillna(-1, inplace=True)  # Placeholder for predictions
        X_task = task.drop(["Rating"], axis=1).values
        logging.info("Data preprocessing completed successfully.")
        assert all(map(lambda m: isinstance(m, Movie), X_train)), "X_train does not contain objects of type Movie"
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return

    # Train Decision Tree
    logging.info("Training Decision Tree model...")
    try:
        tree = DecisionTree(max_depth=10)
        tree.fit(X_train, y_train)
        logging.info("Decision Tree model trained successfully.")
    except Exception as e:
        logging.error(f"Error training Decision Tree: {e}")
        return

    # Predict for task data
    logging.info("Making predictions for task data...")
    try:
        task["Rating"] = tree.predict(X_task)
        logging.info("Predictions completed successfully.")
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return

    # Save results
    logging.info("Saving results to CSV...")
    try:
        path = "task_with_predictions.csv"
        task.to_csv(path, index=False)
        logging.info(f"Results saved successfully to {path}.")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        return


if __name__ == "__main__":
    main()
