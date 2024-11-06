# for numerical features inverse absolute difference
def numerical_similarity(num1, num2):
    return 1 / (1 + abs(num1 - num2))

# for categorical features - binary match
def categorical_similarity(value1, value2):
    return 1 if value1 == value2 else 0  # 1 if values match, 0 otherwise

# for genres - set-based Feature
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
