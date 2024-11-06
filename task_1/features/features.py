import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from task_1.tmdb.client import Client


def create_feature_dataframe(movie_ids, client):
    data = []
    for movie_id in movie_ids:
        movie_data = client.fetch_movie_data(movie_id)
        features = client.extract_features(movie_data)
        features['movie_id'] = movie_id
        data.append(features)
    df = pd.DataFrame(data)
    df.set_index('movie_id', inplace=True)

    return df

    # df_normalized = preprocess_and_normalize_data(df)
    #
    # return df_normalized

def preprocess_and_normalize_data(df: pd.DataFrame):
    categorical_columns = ['genre', 'director', 'cast', 'language', 'country_of_origin']
    numerical_columns = ['rating', 'budget', 'revenue', 'runtime']

    # handling missing values in numerical columns with mean
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    numerical_transformer = MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numerical_transformer, numerical_columns)
        ]
    )

    transformed_data = preprocessor.fit_transform(df)

    column_names = (preprocessor.transformers_[0][1].get_feature_names_out(categorical_columns).tolist() +
                    numerical_columns)

    df_normalized = pd.DataFrame(transformed_data, columns=column_names, index=df.index)

    return df_normalized


# Example usage for dataframe:
if __name__ == '__main__':
    client = Client()
    movie_ids = [550, 500, 600]
    feature_df = create_feature_dataframe(movie_ids, client)
    print(feature_df.head())