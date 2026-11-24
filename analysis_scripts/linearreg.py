
from pathlib import Path

import pandas as pd



# 1. LOAD DATASET


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "hotel_accounts_analysis_ready.xlsx"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    DATA_FILE,
    sheet_name="Regression_Encoded"
)

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================================
# 2. HANDLE MISSING VALUES


def calculate_mean(values):

    total = 0

    for value in values:
        total += value

    return total / len(values)


def calculate_median(values):

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n % 2 == 0:
        return (
            sorted_values[n//2 - 1] +
            sorted_values[n//2]
        ) / 2

    return sorted_values[n//2]


def handle_missing_values(data):

    columns = len(data[0])

    for j in range(columns):

        values = []

        for i in range(len(data)):
            if data[i][j] is not None:
                values.append(data[i][j])

        median = calculate_median(values)

        for i in range(len(data)):
            if data[i][j] is None:
                data[i][j] = median

    return data


# ==========================================
# 3. SEPARATE INDEPENDENT AND TARGET VARIABLES
# ==========================================

target = "Transaction_Amount_INR"

feature_names = [
    col for col in df.columns
    if col != target
]

X = df[feature_names].values.tolist()
Y = df[target].tolist()

X = handle_missing_values(X)

print("\nIndependent Features:", feature_names)


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

def train_test_split(X, Y):

    X_train = []
    X_test = []

    Y_train = []
    Y_test = []

    for i in range(len(X)):

        if i % 5 == 0:

            X_test.append(X[i])
            Y_test.append(Y[i])

        else:

            X_train.append(X[i])
            Y_train.append(Y[i])

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 5. STANDARDIZE INDEPENDENT VARIABLES
# ==========================================

def calculate_std(values, mean):

    total = 0

    for value in values:
        total += (value - mean) ** 2

    variance = total / len(values)

    return variance ** 0.5


def standardize_train(X_train):

    means = []
    stds = []

    for j in range(len(X_train[0])):

        column = []

        for i in range(len(X_train)):
            column.append(X_train[i][j])

        mean = calculate_mean(column)
        std = calculate_std(column, mean)

        means.append(mean)
        stds.append(std)

    standardized = []

    for row in X_train:

        new_row = []

        for j in range(len(row)):

            if stds[j] == 0:
                new_row.append(0)
            else:
                new_row.append(
                    (row[j] - means[j]) / stds[j]
                )

        standardized.append(new_row)

    return standardized, means, stds


def standardize_test(X_test, means, stds):

    result = []

    for row in X_test:

        new_row = []

        for j in range(len(row)):

            if stds[j] == 0:
                new_row.append(0)
            else:
                new_row.append(
                    (row[j] - means[j]) / stds[j]
                )

        result.append(new_row)

    return result


X_train_scaled, X_means, X_stds = standardize_train(X_train)

X_test_scaled = standardize_test(
    X_test, X_means, X_stds
)


# ==========================================
# 6. STANDARDIZE TARGET VARIABLE
# ==========================================

Y_mean = calculate_mean(Y_train)

Y_std = calculate_std(Y_train, Y_mean)

Y_train_scaled = []

for value in Y_train:

    if Y_std == 0:
        Y_train_scaled.append(0)
    else:
        Y_train_scaled.append(
            (value - Y_mean) / Y_std
        )


# ==========================================
# 7. ADD INTERCEPT COLUMN
# ==========================================

def add_intercept(X):

    result = []

    for row in X:
        result.append([1.0] + row)

    return result


X_train_scaled = add_intercept(X_train_scaled)
X_test_scaled = add_intercept(X_test_scaled)


# ==========================================
# 8. LINEAR REGRESSION USING GRADIENT DESCENT
# ==========================================

def linear_regression(X, Y, learning_rate, iterations):

    n = len(X)
    features = len(X[0])

    weights = [0.0] * features

    for iteration in range(iterations):

        gradients = [0.0] * features

        for i in range(n):

            prediction = 0

            for j in range(features):
                prediction += X[i][j] * weights[j]

            error = prediction - Y[i]

            for j in range(features):
                gradients[j] += error * X[i][j]

        for j in range(features):

            gradients[j] = gradients[j] / n

            weights[j] -= (
                learning_rate * gradients[j]
            )

    return weights


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

weights = linear_regression(
    X_train_scaled,
    Y_train_scaled,
    learning_rate=0.01,
    iterations=10000
)

print("\nModel Weights:")
print(weights)


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

def predict(X, weights):

    predictions = []

    for row in X:

        prediction = 0

        for j in range(len(weights)):
            prediction += row[j] * weights[j]

        predictions.append(prediction)

    return predictions


Y_pred_scaled = predict(
    X_test_scaled, weights
)


# Convert predictions back to original INR scale

Y_pred = []

for value in Y_pred_scaled:
    Y_pred.append(value * Y_std + Y_mean)


# 11. CALCULATE MAE

def calculate_mae(actual, predicted):

    total = 0

    for i in range(len(actual)):
        total += abs(actual[i] - predicted[i])

    return total / len(actual)


# 12. CALCULATE MSE


def calculate_mse(actual, predicted):

    total = 0

    for i in range(len(actual)):
        total += (actual[i] - predicted[i]) ** 2

    return total / len(actual)



# 13. CALCULATE RMSE

def calculate_rmse(actual, predicted):

    mse = calculate_mse(actual, predicted)

    return mse ** 0.5


# 14. CALCULATE R-SQUARED


def calculate_r2(actual, predicted):

    mean = calculate_mean(actual)

    ss_total = 0
    ss_residual = 0

    for i in range(len(actual)):

        ss_total += (actual[i] - mean) ** 2

        ss_residual += (
            actual[i] - predicted[i]
        ) ** 2

    if ss_total == 0:
        return 0

    return 1 - (ss_residual / ss_total)


# 15. MODEL EVALUATION

mae = calculate_mae(Y_test, Y_pred)

mse = calculate_mse(Y_test, Y_pred)

rmse = calculate_rmse(Y_test, Y_pred)

r2 = calculate_r2(Y_test, Y_pred)

print("\nMODEL PERFORMANCE")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)



# 16. CONVERT WEIGHTS TO ORIGINAL SCALE

coefficients = []

for j in range(len(feature_names)):

    if X_stds[j] == 0:
        coefficient = 0
    else:
        coefficient = (
            weights[j + 1] * Y_std / X_stds[j]
        )

    coefficients.append(coefficient)


intercept = Y_mean

for j in range(len(feature_names)):
    intercept -= coefficients[j] * X_means[j]

print("\nREGRESSION EQUATION")

print("Intercept:", intercept)

for j in range(len(feature_names)):
    print(feature_names[j], ":", coefficients[j])


# 17. CREATE PREDICTION DATAFRAME

results = pd.DataFrame({

    "Actual_Amount": Y_test,
    "Predicted_Amount": Y_pred

})


# 18. SAVE RESULTS

coefficient_df = pd.DataFrame({

    "Feature": ["Intercept"] + feature_names,

    "Coefficient": [intercept] + coefficients

})

metrics_df = pd.DataFrame({

    "Metric": ["MAE", "MSE", "RMSE", "R2"],

    "Value": [mae, mse, rmse, r2]

})


with pd.ExcelWriter(OUTPUT_FOLDER / "Linear_Regression_Results.xlsx") as writer:

    results.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    coefficient_df.to_excel(
        writer,
        sheet_name="Coefficients",
        index=False
    )

    metrics_df.to_excel(
        writer,
        sheet_name="Model_Evaluation",
        index=False
    )


print("\nLinear Regression Completed Successfully!")
