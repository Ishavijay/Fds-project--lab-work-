# Unit IV: feature engineering, scaling, regression, and optimization.


def make_category_columns(data, column):
    # Return the table with one indicator column for each category value.
    import pandas as pd

    categories = []

    for value in data[column]:
        if value not in categories:
            categories.append(value)

    encoded_columns = {}
    for existing_column in data.columns:
        if existing_column != column:
            encoded_columns[existing_column] = list(data[existing_column])

    for category in categories:
        new_column = f"{column}_{category}"
        encoded_columns[new_column] = [
            1 if value == category else 0 for value in data[column]
        ]

    return pd.DataFrame(encoded_columns, index=data.index)


def scale_numbers(data):
    # Return numeric data standardized to a common scale.
    import numpy as np

    if hasattr(data, "columns"):
        rows = []
        for row_index in range(len(data)):
            row = []
            for column_index in range(len(data.columns)):
                row.append(data.iat[row_index, column_index])
            rows.append(row)
    else:
        rows = [list(row) for row in data]

    if not rows:
        return np.array([])

    column_count = len(rows[0])
    means = []
    standard_deviations = []
    for column_index in range(column_count):
        values = [row[column_index] for row in rows]
        total = 0
        for value in values:
            total += value
        mean = total / len(values)

        squared_total = 0
        for value in values:
            squared_total += (value - mean) ** 2
        variance = squared_total / len(values)
        means.append(mean)
        standard_deviations.append(variance**0.5)

    scaled_rows = []
    for row in rows:
        scaled_row = []
        for column_index, value in enumerate(row):
            deviation = standard_deviations[column_index]
            scaled_value = (
                0
                if deviation == 0
                else (value - means[column_index]) / deviation
            )
            scaled_row.append(scaled_value)

        scaled_rows.append(scaled_row)

    return np.array(scaled_rows)


def fit_linear_model(features, answers):
    # Fit and return a linear regression model for the supplied examples.
    rows = _rows_from_table(features)
    if not rows or len(rows) != len(answers):
        raise ValueError("Features and answers must have the same non-zero length.")

    feature_count = len(rows[0])
    augmented_rows = [[1.0] + [float(value) for value in row] for row in rows]
    normal_matrix = []
    target_vector = []

    for left_index in range(feature_count + 1):
        normal_row = []
        for right_index in range(feature_count + 1):
            total = 0
            for row in augmented_rows:
                total += row[left_index] * row[right_index]
            normal_row.append(total)
        normal_matrix.append(normal_row)
        target_total = 0
        for row, answer in zip(augmented_rows, answers):
            target_total += row[left_index] * float(answer)
        target_vector.append(target_total)

    parameters = _solve_linear_system(normal_matrix, target_vector)

    return _ManualLinearModel(parameters[1:], parameters[0])


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


def _solve_linear_system(matrix, values):
    # Solve a square system using Gaussian elimination with pivoting.
    size = len(values)
    augmented = [matrix[row][:] + [values[row]] for row in range(size)]

    for pivot_index in range(size):
        best_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        augmented[pivot_index], augmented[best_row] = (
            augmented[best_row],
            augmented[pivot_index],
        )
        pivot = augmented[pivot_index][pivot_index]
        if abs(pivot) < 1e-12:
            raise ValueError("The feature data does not have a unique solution.")

        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= (
                    factor * augmented[pivot_index][column_index]
                )

    return [augmented[row_index][size] for row_index in range(size)]


class _ManualLinearModel:
    # Store fitted coefficients and provide prediction and score operations.

    def __init__(self, coefficients, intercept):
        self.coef_ = coefficients
        self.intercept_ = intercept

    def predict(self, features):
        predictions = []
        for row in _rows_from_table(features):
            prediction = self.intercept_
            for coefficient, value in zip(self.coef_, row):
                prediction += coefficient * value

            predictions.append(prediction)
        return predictions

    def score(self, features, answers):
        predictions = self.predict(features)
        answer_total = 0
        for answer in answers:
            answer_total += answer
        answer_mean = answer_total / len(answers)

        residual_total = 0
        for actual, predicted in zip(answers, predictions):
            residual_total += (actual - predicted) ** 2
        total = 0
        for actual in answers:
            total += (actual - answer_mean) ** 2

        if total == 0:
            return 1.0 if residual_total == 0 else 0.0
        return 1 - residual_total / total


def gradient_descent(gradient, start, step_size=0.1, steps=100):
    # Minimize a function by repeatedly following its gradient downhill.
    # The gradient returns the slope; step_size sets each move and steps sets
    # the number of updates.
    number = start
    for _ in range(steps):
        number -= step_size * gradient(number)
    return number

