
import pandas as pd
import math


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel(
    "hotel_accounts_analysis_ready(4).xlsx",
    sheet_name="Regression_Encoded"
)

# Target variable
y = df["Is_Expense"].tolist()

# Remove target from input features
feature_names = [
    col for col in df.columns
    if col != "Is_Expense"
]

X = df[feature_names].values.tolist()

print("Dataset Shape:", len(X), len(X[0]))


# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================

def calculate_median(values):

    values = sorted(values)
    n = len(values)

    if n % 2 == 0:
        return (values[n//2 - 1] + values[n//2]) / 2

    return values[n//2]


def handle_missing_values(X):

    columns = len(X[0])

    for j in range(columns):

        values = []

        for i in range(len(X)):
            if X[i][j] is not None:
                values.append(X[i][j])

        median = calculate_median(values)

        for i in range(len(X)):
            if X[i][j] is None:
                X[i][j] = median

    return X


X = handle_missing_values(X)


# ==========================================
# 3. CALCULATE MEAN
# ==========================================

def calculate_mean(data):

    means = []

    for j in range(len(data[0])):

        total = 0

        for i in range(len(data)):
            total += data[i][j]

        means.append(total / len(data))

    return means


# ==========================================
# 4. CALCULATE STANDARD DEVIATION
# ==========================================

def calculate_std(data, means):

    stds = []

    for j in range(len(data[0])):

        total = 0

        for i in range(len(data)):
            total += (data[i][j] - means[j]) ** 2

        stds.append(math.sqrt(total / len(data)))

    return stds


# ==========================================
# 5. STANDARDIZE DATA
# ==========================================

def standardize_data(data, means, stds):

    result = []

    for row in data:

        new_row = []

        for j in range(len(row)):

            if stds[j] == 0:
                new_row.append(0)
            else:
                value = (row[j] - means[j]) / stds[j]
                new_row.append(value)

        result.append(new_row)

    return result


means = calculate_mean(X)
stds = calculate_std(X, means)

X_scaled = standardize_data(X, means, stds)


# ==========================================
# 6. SEPARATE DATA INTO TWO CLASSES
# ==========================================

def separate_classes(X, y):

    class_0 = []
    class_1 = []

    for i in range(len(X)):

        if y[i] == 0:
            class_0.append(X[i])

        else:
            class_1.append(X[i])

    return class_0, class_1


class_0, class_1 = separate_classes(X_scaled, y)


# ==========================================
# 7. CALCULATE CLASS MEANS
# ==========================================

mean_0 = calculate_mean(class_0)
mean_1 = calculate_mean(class_1)


# ==========================================
# 8. CALCULATE WITHIN-CLASS SCATTER MATRIX
# ==========================================

def within_class_scatter(classes, means):

    features = len(means)

    scatter = [
        [0.0 for j in range(features)]
        for i in range(features)
    ]

    for data, mean in zip(classes, means):

        for row in data:

            for i in range(features):

                for j in range(features):

                    scatter[i][j] += (
                        (row[i] - mean[i]) *
                        (row[j] - mean[j])
                    )

    return scatter


Sw = within_class_scatter(
    [class_0, class_1],
    [mean_0, mean_1]
)


# ==========================================
# 9. ADD REGULARIZATION
# ==========================================

def regularize_matrix(matrix):

    n = len(matrix)

    diagonal_mean = 0

    for i in range(n):
        diagonal_mean += matrix[i][i]

    diagonal_mean /= n

    ridge = max(diagonal_mean * 0.000001, 0.000001)

    for i in range(n):
        matrix[i][i] += ridge

    return matrix


Sw = regularize_matrix(Sw)


# ==========================================
# 10. MATRIX INVERSE
# GAUSS-JORDAN ELIMINATION
# ==========================================

def matrix_inverse(matrix):

    n = len(matrix)

    identity = []

    for i in range(n):

        row = []

        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)

        identity.append(row)

    augmented = []

    for i in range(n):
        augmented.append(matrix[i][:] + identity[i])

    for i in range(n):

        # Partial pivoting
        max_row = i

        for k in range(i + 1, n):

            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k

        augmented[i], augmented[max_row] = (
            augmented[max_row],
            augmented[i]
        )

        pivot = augmented[i][i]

        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular")

        for j in range(2 * n):
            augmented[i][j] /= pivot

        for k in range(n):

            if k != i:

                factor = augmented[k][i]

                for j in range(2 * n):

                    augmented[k][j] -= (
                        factor * augmented[i][j]
                    )

    inverse = []

    for i in range(n):
        inverse.append(augmented[i][n:])

    return inverse


# ==========================================
# 11. MATRIX VECTOR MULTIPLICATION
# ==========================================

def matrix_vector_multiply(matrix, vector):

    result = []

    for i in range(len(matrix)):

        total = 0

        for j in range(len(vector)):
            total += matrix[i][j] * vector[j]

        result.append(total)

    return result


# ==========================================
# 12. CALCULATE LDA DISCRIMINANT VECTOR
# ==========================================

def calculate_lda(Sw, mean_0, mean_1):

    inverse_Sw = matrix_inverse(Sw)

    mean_difference = []

    for i in range(len(mean_0)):

        mean_difference.append(
            mean_1[i] - mean_0[i]
        )

    weights = matrix_vector_multiply(
        inverse_Sw,
        mean_difference
    )

    return weights


weights = calculate_lda(
    Sw,
    mean_0,
    mean_1
)


# ==========================================
# 13. PROJECT DATA ON LDA AXIS
# ==========================================

def lda_transform(data, weights):

    result = []

    for row in data:

        total = 0

        for j in range(len(row)):
            total += row[j] * weights[j]

        result.append(total)

    return result


lda_scores = lda_transform(
    X_scaled,
    weights
)


# ==========================================
# 14. CALCULATE CLASS THRESHOLD
# ==========================================

def calculate_threshold(class_0, class_1, weights):

    projected_0 = lda_transform(class_0, weights)
    projected_1 = lda_transform(class_1, weights)

    mean_projected_0 = sum(projected_0) / len(projected_0)
    mean_projected_1 = sum(projected_1) / len(projected_1)

    threshold = (
        mean_projected_0 + mean_projected_1
    ) / 2

    return threshold


threshold = calculate_threshold(
    class_0,
    class_1,
    weights
)


# ==========================================
# 15. CLASSIFY TRANSACTIONS
# ==========================================

def predict_classes(scores, threshold):

    predictions = []

    for score in scores:

        if score >= threshold:
            predictions.append(1)

        else:
            predictions.append(0)

    return predictions


predictions = predict_classes(
    lda_scores,
    threshold
)


# ==========================================
# 16. CALCULATE ACCURACY
# ==========================================

def calculate_accuracy(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i] == predicted[i]:
            correct += 1

    return (correct / len(actual)) * 100


accuracy = calculate_accuracy(y, predictions)


# ==========================================
# 17. DISPLAY RESULTS
# ==========================================

print("\nLDA Results")

print("Original Features:", len(feature_names))
print("Reduced Features: 1")

print("\nLDA Discriminant Weights:")
for i in range(len(feature_names)):
    print(feature_names[i], ":", weights[i])

print("\nLDA Threshold:", threshold)

print("\nClassification Accuracy:", accuracy, "%")


# ==========================================
# 18. CREATE RESULT DATAFRAME
# ==========================================

result_df = pd.DataFrame({

    "Actual_Class": y,
    "LDA_Score": lda_scores,
    "Predicted_Class": predictions

})


# ==========================================
# 19. SAVE RESULTS
# ==========================================

weights_df = pd.DataFrame({

    "Feature": feature_names,
    "LDA_Weight": weights

})

with pd.ExcelWriter("Manual_LDA_Results.xlsx") as writer:

    result_df.to_excel(
        writer,
        sheet_name="LDA_Results",
        index=False
    )

    weights_df.to_excel(
        writer,
        sheet_name="LDA_Weights",
        index=False
    )

print("\nLDA completed successfully!")
