# Unit V: beginner-friendly machine learning helpers.
# Functions train classifiers, make predictions, group data, and find frequent items.


def train_classifier(features, answers):
    # Train and return a random forest using labeled examples.
    if len(features) == 0 or len(features) != len(answers):
        raise ValueError("Features and answers must have the same non-zero length.")

    model = _ManualRandomForest(seed=42, tree_count=7)
    model.fit(features, list(answers))

    return model


def make_predictions(model, new_features):
    # Use a trained model to predict labels for new examples.
    predictions = []
    for row in _rows_from_table(new_features):
        votes = {}
        for tree in model.trees:
            while isinstance(tree, tuple):
                feature_index, threshold, left_tree, right_tree = tree
                if row[feature_index] <= threshold:
                    tree = left_tree
                else:
                    tree = right_tree
            votes[tree] = votes.get(tree, 0) + 1

        predictions.append(max(votes, key=votes.get))

    return predictions


def check_accuracy(model, features, answers):
    # Return the fraction of examples the model classifies correctly.
    predictions = make_predictions(model, features)
    correct = 0
    for prediction, answer in zip(predictions, answers):
        if prediction == answer:
            correct += 1

    return correct / len(answers)


def make_clusters(data, number_of_groups=3):
    # Assign each row in the data to one of the requested clusters.
    rows = [list(row) for row in data]
    if not rows or number_of_groups < 1 or number_of_groups > len(rows):
        raise ValueError("Choose a cluster count between one and the row count.")

    centers = [row[:] for row in rows[:number_of_groups]]
    labels = [0] * len(rows)

    for _ in range(100):
        changed = False
        for row_index, row in enumerate(rows):
            best_cluster = min(
                range(number_of_groups),
                key=lambda cluster_index: _squared_distance(
                    row,
                    centers[cluster_index],
                ),
            )
            if labels[row_index] != best_cluster:
                labels[row_index] = best_cluster
                changed = True

        new_centers = []
        for cluster_index in range(number_of_groups):
            members = [
                row for row_index, row in enumerate(rows)
                if labels[row_index] == cluster_index
            ]
            if not members:
                new_centers.append(centers[cluster_index])
                continue
            center = []
            for column_index in range(len(rows[0])):
                total = 0
                for row in members:
                    total += row[column_index]
                center.append(total / len(members))

            new_centers.append(center)

        centers = new_centers
        if not changed:
            break

    return labels


def find_frequent_items(baskets, minimum_fraction=0.5):
    # Return frequent single items and pairs with their support fractions.
    if not baskets:
        raise ValueError("Give at least one basket.")
    if not 0 < minimum_fraction <= 1:
        raise ValueError(
            "minimum_fraction must be above zero and at most one."
        )
    counts = {}
    for basket in baskets:
        items = sorted(set(basket))

        for item_index, item in enumerate(items):
            single = (item,)
            counts[single] = counts.get(single, 0) + 1
            for other_index in range(item_index + 1, len(items)):
                pair = (item, items[other_index])
                counts[pair] = counts.get(pair, 0) + 1

    return {
        item: count / len(baskets)
        for item, count in counts.items()
        if count / len(baskets) >= minimum_fraction
    }


def _squared_distance(first, second):
    # Return squared Euclidean distance between two rows.
    total = 0
    for first_value, second_value in zip(first, second):
        total += (first_value - second_value) ** 2

    return total


class _ManualRandomForest:
    # Build a small random forest from decision trees and Python loops.

    def __init__(self, seed=42, tree_count=7, max_depth=8):
        self.seed = seed
        self.tree_count = tree_count
        self.max_depth = max_depth
        self.trees = []
        self.classes = []

    def fit(self, features, answers):
        import random

        rng = random.Random(self.seed)
        self.classes = []
        for answer in answers:
            if answer not in self.classes:
                self.classes.append(answer)

        feature_rows = _rows_from_table(features)
        self.trees = []

        for _ in range(self.tree_count):
            sample_indices = [rng.randrange(len(feature_rows)) for _ in feature_rows]
            sample_rows = [feature_rows[index] for index in sample_indices]
            sample_answers = [answers[index] for index in sample_indices]
            self.trees.append(
                self._build_tree(sample_rows, sample_answers, 0, rng)
            )
        return self

    def _build_tree(self, rows, answers, depth, rng):
        if depth >= self.max_depth or len(set(answers)) == 1 or len(rows) < 2:
            return self._majority_label(answers)

        feature_count = len(rows[0])
        feature_indices = list(range(feature_count))
        rng.shuffle(feature_indices)
        feature_indices = feature_indices[: max(1, int(feature_count**0.5))]
        best_split = None
        best_impurity = self._gini(answers)

        for feature_index in feature_indices:
            values = sorted(set(row[feature_index] for row in rows))
            for lower, upper in zip(values, values[1:]):
                threshold = (lower + upper) / 2
                left_indices = [
                    index for index, row in enumerate(rows)
                    if row[feature_index] <= threshold
                ]
                right_indices = [
                    index for index, row in enumerate(rows)
                    if row[feature_index] > threshold
                ]
                if not left_indices or not right_indices:
                    continue
                impurity = (
                    len(left_indices) * self._gini([answers[i] for i in left_indices])
                    + len(right_indices)
                    * self._gini([answers[i] for i in right_indices])
                ) / len(rows)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split = (feature_index, threshold, left_indices, right_indices)

        if best_split is None:
            return self._majority_label(answers)

        feature_index, threshold, left_indices, right_indices = best_split
        return (
            feature_index,
            threshold,
            self._build_tree(
                [rows[i] for i in left_indices],
                [answers[i] for i in left_indices],
                depth + 1,
                rng,
            ),
            self._build_tree(
                [rows[i] for i in right_indices],
                [answers[i] for i in right_indices],
                depth + 1,
                rng,
            ),
        )

    def _gini(self, answers):
        counts = {}
        for answer in answers:
            counts[answer] = counts.get(answer, 0) + 1

        total = len(answers)
        impurity = 1
        for count in counts.values():
            impurity -= (count / total) ** 2
        return impurity

    def _majority_label(self, answers):
        counts = {}
        for answer in answers:
            counts[answer] = counts.get(answer, 0) + 1

        return max(counts, key=counts.get)

    def _predict_tree(self, tree, row):
        while isinstance(tree, tuple):
            feature_index, threshold, left_tree, right_tree = tree
            tree = left_tree if row[feature_index] <= threshold else right_tree

        return tree

    def predict(self, features):
        predictions = []
        for row in _rows_from_table(features):
            votes = {}
            for tree in self.trees:
                label = self._predict_tree(tree, row)
                votes[label] = votes.get(label, 0) + 1

            predictions.append(max(votes, key=votes.get))

        return predictions

    def score(self, features, answers):
        predictions = self.predict(features)
        correct = 0
        for prediction, answer in zip(predictions, answers):
            if prediction == answer:
                correct += 1

        return correct / len(answers)


def _rows_from_table(data):
    # Copy rows out of a DataFrame or nested list with explicit loops.
    if hasattr(data, "columns"):
        rows = []
        for row_index in range(len(data)):
            row = []
            for column_index in range(len(data.columns)):
                row.append(data.iat[row_index, column_index])
            rows.append(row)

        return rows
    return [list(row) for row in data]

