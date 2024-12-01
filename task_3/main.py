from data.movie import train, task
from task_3.cross_validation import kfold_cross_validation
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function

# Pivot table: rows = UserID, columns = MovieID, values = Rating
rating_matrix = train.pivot(index='UserID', columns='MovieID', values='Rating')

# Precompute similarities and cache them
similarity_cache = {}

def main():
    print("Starting k-Fold Cross-Validation...")
    mean_mse, std_mse = kfold_cross_validation(train, similarity_function, similarity_cache)
    print(f"Mean MSE: {mean_mse:.4f}, Std MSE: {std_mse:.4f}")

    # Predict ratings for task.csv
    predicted_ratings = []
    for _, row in task.iterrows():
        target_user = row['UserID']
        target_movie = row['MovieID']
        predicted_rating = predict_rating(target_user, target_movie, similarity_function, similarity_cache, rating_matrix)
        predicted_ratings.append(round(predicted_rating))

    # Replace NULL values in task.csv with predicted ratings
    task['Rating'] = predicted_ratings
    task.to_csv('submission.csv', index=False, sep=';')

    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()