from pathlib import Path
import logging

from data.movie import task

_RESULT_FILE = Path('task_1_result.csv').absolute()


def predict_score(user_id: int, movie_id: int) -> int:
    # todo: provide actual implementation
    return user_id - movie_id


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    logging.info("Calculating predictions...")
    # apply the prediction function to each row in the task dataframe
    predicted = task.apply(lambda row: predict_score(row['UserID'], row['MovieID']), axis=1).astype(int)

    task_with_predictions = task.copy()
    task_with_predictions['Rating'] = predicted

    task_with_predictions.to_csv(_RESULT_FILE, index=False, sep=';')
    logging.info("Written results file to %s", _RESULT_FILE)


if __name__ == '__main__':
    main()
