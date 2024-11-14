import graphviz
from sklearn.tree import export_graphviz


def visualize_decision_tree(dt_model, feature_names):
    """
    Visualize the decision tree using Graphviz.

    Parameters:
        dt_model (DecisionTreeClassifier): A trained decision tree model.
        feature_names (list): List of feature names used in the model.

    Returns:
        graphviz.Source: The visualized decision tree.
    """
    dot_data = export_graphviz(dt_model, out_file=None,
                               feature_names=feature_names,
                               class_names=['0', '1', '2', '3', '4', '5'],
                               filled=True, rounded=True,
                               special_characters=True)

    graph = graphviz.Source(dot_data)
    return graph
