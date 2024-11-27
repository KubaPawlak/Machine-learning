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