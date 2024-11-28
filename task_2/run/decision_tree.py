import logging

from movie import Movie
from task_2.classification.decision_tree import DecisionTree
from task_2.util import SubmissionGenerator


class DecisionTreeSubmissionGenerator(SubmissionGenerator):
    def create_fitted_classifier(self, movies: list[Movie], labels: list[int]) -> DecisionTree:
        tree = DecisionTree(max_depth=2)
        tree.fit(movies, labels)
        return tree

    def predict(self, classifier: DecisionTree, movies: list[Movie]) -> list[int]:
        assert isinstance(classifier, DecisionTree)
        return classifier.predict(movies)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    submission_generator = DecisionTreeSubmissionGenerator('submission_tree.csv')
    submission_generator.run()
