import logging

from movie import Movie
from task_2.classification.decision_tree import DecisionTree
from task_2.classification.random_forest import RandomForestClassifier
from task_2.util import SubmissionGenerator


class DecisionTreeSubmissionGenerator(SubmissionGenerator):
    def create_fitted_classifier(self, movies: list[Movie], labels: list[int]) -> DecisionTree | RandomForestClassifier:
        tree = DecisionTree(max_depth=10)
        tree.fit(movies, labels)
        return tree

    def predict(self, classifier: DecisionTree | RandomForestClassifier, movies: list[Movie]) -> list[int]:
        return classifier.predict(movies)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    submission_generator = DecisionTreeSubmissionGenerator('submission_tree.csv')
    submission_generator.run()
