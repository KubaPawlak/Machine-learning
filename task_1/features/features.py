import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

def create_feature_vector(movie_features: dict, fit_scaler=False, fit_encoder=False) -> pd.Series:
    df = pd.DataFrame([movie_features])

    categorical_cols = ['genre', 'director', 'cast', 'language', 'country_of_origin']
    numerical_cols = ['rating', 'budget', 'revenue', 'runtime']

    if fit_encoder:
        encoded_features = encoder.fit_transform(df[categorical_cols])
    else:
        encoded_features = encoder.transform(df[categorical_cols])

    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    combined = pd.concat([encoded_df, df[numerical_cols]], axis=1)

    if fit_scaler:
        combined[numerical_cols] = scaler.fit_transform(combined[numerical_cols])
    else:
        combined[numerical_cols] = scaler.transform(combined[numerical_cols])

    return combined.iloc[0]


def convert_movies_to_feature_vectors(movies_list: list) -> pd.DataFrame:
    feature_vectors = pd.DataFrame()
    for i, movie in enumerate(movies_list):
        fit_scaler = (i == 0)
        fit_encoder = (i == 0)
        features = create_feature_vector(movie, fit_scaler, fit_encoder)
        feature_vectors = feature_vectors._append(features, ignore_index=True)

    return feature_vectors
