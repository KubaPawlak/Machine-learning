import logging
import pathlib

from .classification.decision_tree import DecisionTree


def _visit_node(tree: DecisionTree) -> str:
    node_id = id(tree)
    if tree.leaf_value is not None:
        # is a leaf node
        return f'{node_id} [label="{str(tree.leaf_value)}"]\n'

    # is a choice node
    true_node_id = id(tree.child_success)
    false_node_id = id(tree.child_fail)
    output = ""
    output += f'{node_id} [label="{str(tree.choice)}"]\n'
    output += f'{node_id} -> {true_node_id} [label="true"]\n'
    output += f'{node_id} -> {false_node_id} [label="false"]\n'
    output += _visit_node(tree.child_success)
    output += _visit_node(tree.child_fail)
    return output


def generate_graphviz_file(tree: DecisionTree) -> str:
    """Generates a dot file provided a decision tree"""
    return f'digraph {{\n{_visit_node(tree)}}}'


from main import DecisionTreeModel


def _get_tree_for_user(user_id: int) -> DecisionTree:
    from data.movie import train
    train_user = train[train['UserID'] == user_id]
    tree_model = DecisionTreeModel(train)
    return tree_model.create_model(train_user)


def _main():
    logging.basicConfig(level=logging.INFO)
    dot_file_path = pathlib.Path(__file__).parent / 'submission' / 'tree.dot'
    user_id: int = 1642

    tree = _get_tree_for_user(user_id)
    logging.info("Generating dot file")
    dot = generate_graphviz_file(tree)

    if dot_file_path.exists():
        logging.warning("File %s already exists. It will be overwritten.", str(dot_file_path))
    else:
        dot_file_path.parent.mkdir(parents=True, exist_ok=True)

    dot_file_path.write_text(dot)
    logging.info("Dot file successfully written to %s. Run 'dot -Tpng %s' to generate the image file.",
                 str(dot_file_path), str(dot_file_path))


if __name__ == '__main__':
    _main()
