import logging
from pathlib import Path

from data.movie import train, task
from task_3.cross_validation import kfold_cross_validation
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function

logger = logging.getLogger(__name__)

SUBMISSION_PATH = Path(__file__).parent / 'submission/submission.csv'

def main():
    logging.basicConfig(level=logging.INFO)

    rating_matrix = train.pivot(index='UserID', columns='MovieID', values='Rating')

    predicted_ratings = []
    for _, row in task.iterrows():
        target_user = row['UserID']
        target_movie = row['MovieID']
        predicted_rating = predict_rating(target_user, target_movie, similarity_function, rating_matrix)
        predicted_rating = int(round(predicted_rating))
        assert 0 <= predicted_rating <= 5
        predicted_ratings.append(predicted_rating)

    # Replace null values in task.csv with predicted ratings
    task['Rating'] = predicted_ratings
    assert task['Rating'].isna().sum() == 0

    SUBMISSION_PATH.parent.mkdir(parents=True,exist_ok=True)
    task.to_csv(SUBMISSION_PATH, index=False, sep=';')

    logger.info(f"Predictions saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()