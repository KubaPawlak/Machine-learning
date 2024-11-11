import numpy as np

from task_1.classifier import KNeighborsClassifier
from task_1.features import cosine_similarity
from task_1.tmdb.client import Client


# def main()-> None:
#     client = Client()
#     movie = client.get_movie(62)
#     print(movie)
#
# if __name__ == '__main__':
#     main()

def main() -> None:
    client = Client()

    movie_ids = [62, 63, 64]
    movies = [client.get_movie(movie_id) for movie_id in movie_ids]

    # Feature vectors for the fetched movies
    features = np.array([movie.to_feature_vector() for movie in movies])

    # Example ratings for the movies
    labels = np.array([4, 5, 3])

    # Train classifier
    knn = KNeighborsClassifier(n_neighbors=2, similarity_function=cosine_similarity)

    # Print
    for i, movie_id in enumerate(movie_ids):
        prediction = knn.fit_predict(features, labels, features[i])  # Predict for each movie
        print(f"Predicted rating for movie {movie_id}: {prediction}")


if __name__ == '__main__':
    main()