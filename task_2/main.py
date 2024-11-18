import logging

from data.movie import train, task
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
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return

    # Train Decision Tree
    logging.info("Training Decision Tree model...")
    try:
        tree = DecisionTree(max_depth=10, min_samples_split=2)
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
        task.to_csv("task_with_predictions.csv", index=False)
        logging.info("Results saved successfully to 'task_with_predictions.csv'.")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        return


if __name__ == "__main__":
    main()