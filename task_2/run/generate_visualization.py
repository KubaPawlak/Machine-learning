import logging
from pathlib import Path

import numpy as np

from task_2.classification.decision_tree import DecisionTree
from task_2.visualisation import generate_graphviz_file

from task_2.decision_tree_main import get_movie_features, train_decision_tree

from data.movie import train

DOT_FILE_PATH = Path(__file__).parent.absolute() / 'tree.dot'
DEFAULT_USER_ID: int = 1642


def _generate_tree_for_user(user_id: int = DEFAULT_USER_ID) -> DecisionTree:
    user_watched_movies = train[train['UserID'] == user_id]
    movie_features = get_movie_features(np.unique(user_watched_movies['MovieID']))
    tree = train_decision_tree(user_watched_movies, movie_features)
    return tree

def _main():
    logging.basicConfig(level=logging.INFO)
    tree = _generate_tree_for_user()
    logging.info("Generating dot file")
    dot = generate_graphviz_file(tree)

    if DOT_FILE_PATH.exists():
        logging.warning("File %s already exists. It will be overwritten.", str(DOT_FILE_PATH))

    DOT_FILE_PATH.write_text(dot)
    logging.info("Dot file successfully written to %s. Run 'dot -Tpng %s' to generate the image file.",
                 str(DOT_FILE_PATH), str(DOT_FILE_PATH))

if __name__ == '__main__':
    _main()

