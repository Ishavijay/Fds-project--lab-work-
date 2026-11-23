"""Simple machine learning helpers for Unit V."""


# Trains a random forest to sort examples into known groups.
def train_classifier(features, answers):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as error:
        raise ImportError(
            "Install scikit-learn first: python -m pip install scikit-learn"
        ) from error
    model = RandomForestClassifier(random_state=42)
    model.fit(features, answers)
    return model


# Uses a trained model to guess the group for new examples.
def make_predictions(model, new_features):
    return model.predict(new_features)


# Tells you what fraction of the guesses were correct.
def check_accuracy(model, features, answers):
    return model.score(features, answers)


# Groups similar examples together with the K-Means method.
def make_clusters(data, number_of_groups=3):
    try:
        from sklearn.cluster import KMeans
    except ImportError as error:
        raise ImportError(
            "Install scikit-learn first: python -m pip install scikit-learn"
        ) from error
    model = KMeans(n_clusters=number_of_groups, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    return labels


# Finds common single items and pairs in shopping baskets.
def find_frequent_items(baskets, minimum_fraction=0.5):
    if not baskets:
        raise ValueError("Give at least one basket.")
    if not 0 < minimum_fraction <= 1:
        raise ValueError(
            "minimum_fraction must be above zero and at most one."
        )
    from itertools import combinations

    counts = {}
    for basket in baskets:
        items = sorted(set(basket))
        for size in (1, 2):
            for group in combinations(items, size):
                counts[group] = counts.get(group, 0) + 1

    return {
        item: count / len(baskets)
        for item, count in counts.items()
        if count / len(baskets) >= minimum_fraction
    }

