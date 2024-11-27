import logging

from task_2.classification.decision_tree import DecisionTree
from task_2.util import get_training_data_for_user, submission_dir
from task_2.visualisation import generate_graphviz_file

DOT_FILE_PATH = submission_dir / 'tree.dot'
DEFAULT_USER_ID: int = 1642

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_tree_for_user(user_id: int = DEFAULT_USER_ID) -> DecisionTree:
    logger.info(f"Generating tree for user {user_id}")
    movies, labels = get_training_data_for_user(user_id)
    tree = DecisionTree(max_depth=10)
    tree.fit(movies, labels)
    return tree


def _main():
    logging.basicConfig(level=logging.INFO)
    tree = _generate_tree_for_user()
    logger.info("Generating dot file")
    dot = generate_graphviz_file(tree)

    if DOT_FILE_PATH.exists():
        logger.warning("File %s already exists. It will be overwritten.", str(DOT_FILE_PATH))

    DOT_FILE_PATH.write_text(dot)
    logger.info("Dot file successfully written to %s. Run 'dot -Tpng %s' to generate the image file.",
                str(DOT_FILE_PATH), str(DOT_FILE_PATH))


if __name__ == '__main__':
    _main()
