# for numerical features: inverse absolute difference
def numerical_similarity(num1, num2):
    return 1 / (1 + abs(num1 - num2))


# for categorical features: binary match
def categorical_similarity(value1, value2):
    return 1 if value1 == value2 else 0


# for genres: Jaccard similarity for set features
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def calculate_similarity(vector_a, vector_b, categorical_cols, numerical_cols, genre_col):
    total_similarity = 0
    total_weight = 0

    for col in numerical_cols:
        sim = numerical_similarity(vector_a[col], vector_b[col])
        total_similarity += sim
        total_weight += 1

    for col in categorical_cols:
        sim = categorical_similarity(vector_a[col], vector_b[col])
        total_similarity += sim
        total_weight += 1

    if genre_col in vector_a and genre_col in vector_b:
        set_a = set(vector_a[genre_col].split(';'))
        set_b = set(vector_b[genre_col].split(';'))
        sim = jaccard_similarity(set_a, set_b)
        total_similarity += sim
        total_weight += 1

    # return average similarity
    return total_similarity / total_weight if total_weight > 0 else 0