import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from data.movie import task, train

train_df = train
user_models = {}

for user_id in train_df['UserID'].unique():
    user_data = train_df[train_df['UserID'] == user_id]
    X = user_data.drop(['UserID', 'MovieID', 'Rating'], axis=1)
    y = user_data['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    user_models[user_id] = dt_model

task_df = task
submission_tree_df = task_df.copy()

for user_id in task_df['UserID'].unique():
    user_data = task_df[task_df['UserID'] == user_id]
    X = user_data.drop(['UserID', 'MovieID', 'Rating'], axis=1)
    dt_model = user_models[user_id]
    predicted_ratings = dt_model.predict(X)
    submission_tree_df.loc[user_data.index, 'Rating'] = predicted_ratings

submission_tree_df.to_csv('submission_tree.csv', index=False)

user_id = 1642
dt_model = user_models[user_id]

dot_data = export_graphviz(dt_model, out_file=None,
                           feature_names=X.columns,
                           class_names=['0', '1', '2', '3', '4', '5'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('tree')
print(graph)
