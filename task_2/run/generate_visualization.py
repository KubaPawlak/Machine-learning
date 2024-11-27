import logging
from pathlib import Path

from task_2.classification.decision_tree import DecisionTree
from task_2.visualisation import generate_graphviz_file

DOT_FILE_PATH = Path(__file__).parent.absolute() / 'tree.dot'
DEFAULT_USER_ID: int = 1642


def generate_tree_for_user(user_id: int = DEFAULT_USER_ID) -> DecisionTree:
    # todo: tree training (imported from another file once it is done)
    logging.info("Generating tree for user %s", user_id)
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tree = generate_tree_for_user()
    logging.info("Generating dot file")
    dot = generate_graphviz_file(tree)

    if DOT_FILE_PATH.exists():
        logging.warning("File %s already exists. It will be overwritten.", str(DOT_FILE_PATH))

    DOT_FILE_PATH.write_text(dot)
    logging.info("Dot file successfully written to %s. Run 'dot -Tpng %s' to generate the image file.",
                 str(DOT_FILE_PATH), str(DOT_FILE_PATH))

