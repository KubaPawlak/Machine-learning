import logging

from movie import Movie
from task_2.classification.random_forest import RandomForestClassifier
from task_2.util import SubmissionGenerator


class RandomForestSubmissionGenerator(SubmissionGenerator):
    def create_fitted_classifier(self, movies: list[Movie], labels: list[int]) ->  RandomForestClassifier:
        tree = RandomForestClassifier(num_trees=10, num_features=3, max_depth=10)
        tree.fit(movies, labels)
        return tree

    def predict(self, classifier: RandomForestClassifier, movies: list[Movie]) -> list[int]:
        assert isinstance(classifier, RandomForestClassifier)
        return classifier.predict(movies)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    submission_generator = RandomForestSubmissionGenerator('submission_forest.csv')
    submission_generator.run()
