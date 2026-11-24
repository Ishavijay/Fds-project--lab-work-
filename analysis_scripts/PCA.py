
from pathlib import Path

import pandas as pd
import math


# ==========================================
# 1. LOAD DATASET
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "hotel_accounts_analysis_ready.xlsx"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    DATA_FILE,
    sheet_name="Regression_Encoded"
)

data = df.values.tolist()
features = df.columns.tolist()

rows = len(data)
cols = len(data[0])

print("Original Shape:", rows, cols)


# ==========================================
# 2. CALCULATE MEAN MANUALLY
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
# 3. CALCULATE STANDARD DEVIATION
# ==========================================

def calculate_std(data, means):

    stds = []

    for j in range(len(data[0])):

        total = 0

        for i in range(len(data)):
            total += (data[i][j] - means[j]) ** 2

        variance = total / len(data)

        stds.append(math.sqrt(variance))

    return stds


# ==========================================
# 4. STANDARDIZE DATA
# ==========================================

def standardize_data(data, means, stds):

    standardized = []

    for i in range(len(data)):

        row = []

        for j in range(len(data[0])):

            if stds[j] == 0:
                row.append(0)
            else:
                value = (data[i][j] - means[j]) / stds[j]
                row.append(value)

        standardized.append(row)

    return standardized


# ==========================================
# 5. TRANSPOSE MATRIX
# ==========================================

def transpose(matrix):

    result = []

    for j in range(len(matrix[0])):

        row = []

        for i in range(len(matrix)):
            row.append(matrix[i][j])

        result.append(row)

    return result


# ==========================================
# 6. MATRIX MULTIPLICATION
# ==========================================

def matrix_multiply(A, B):

    result = []

    for i in range(len(A)):

        row = []

        for j in range(len(B[0])):

            total = 0

            for k in range(len(B)):
                total += A[i][k] * B[k][j]

            row.append(total)

        result.append(row)

    return result


# ==========================================
# 7. CALCULATE COVARIANCE MATRIX
# ==========================================

def covariance_matrix(data):

    n = len(data)
    m = len(data[0])

    means = calculate_mean(data)

    covariance = []

    for i in range(m):

        row = []

        for j in range(m):

            total = 0

            for k in range(n):
                total += (
                    (data[k][i] - means[i]) *
                    (data[k][j] - means[j])
                )

            row.append(total / (n - 1))

        covariance.append(row)

    return covariance


# ==========================================
# 8. MATRIX-VECTOR MULTIPLICATION
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
# 9. VECTOR NORMALIZATION
# ==========================================

def normalize_vector(vector):

    total = 0

    for value in vector:
        total += value * value

    magnitude = math.sqrt(total)

    if magnitude == 0:
        return vector

    return [value / magnitude for value in vector]


# ==========================================
# 10. POWER ITERATION
# ==========================================

def power_iteration(matrix):

    n = len(matrix)

    vector = [1.0] * n
    vector = normalize_vector(vector)

    eigenvalue = 0

    for iteration in range(1000):

        new_vector = matrix_vector_multiply(
            matrix, vector
        )

        new_vector = normalize_vector(new_vector)

        product = matrix_vector_multiply(
            matrix, new_vector
        )

        new_eigenvalue = 0

        for i in range(n):
            new_eigenvalue += new_vector[i] * product[i]

        if abs(new_eigenvalue - eigenvalue) < 1e-10:
            vector = new_vector
            eigenvalue = new_eigenvalue
            break

        vector = new_vector
        eigenvalue = new_eigenvalue

    return eigenvalue, vector


# ==========================================
# 11. DEFLATION
# ==========================================

def deflate_matrix(matrix, eigenvalue, eigenvector):

    n = len(matrix)

    result = []

    for i in range(n):

        row = []

        for j in range(n):

            value = (
                matrix[i][j] -
                eigenvalue * eigenvector[i] * eigenvector[j]
            )

            row.append(value)

        result.append(row)

    return result


# ==========================================
# 12. CALCULATE ALL PRINCIPAL COMPONENTS
# ==========================================

def calculate_pca(covariance, threshold=0.85):

    matrix = [row[:] for row in covariance]

    total_variance = 0

    for i in range(len(matrix)):
        total_variance += matrix[i][i]

    eigenvalues = []
    eigenvectors = []

    cumulative_variance = 0

    for component in range(len(matrix)):

        eigenvalue, eigenvector = power_iteration(matrix)

        if eigenvalue < 0:
            eigenvalue = 0

        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        cumulative_variance += eigenvalue

        if cumulative_variance / total_variance >= threshold:
            break

        matrix = deflate_matrix(
            matrix,
            eigenvalue,
            eigenvector
        )

    return eigenvalues, eigenvectors


# ==========================================
# 13. TRANSFORM DATA INTO PRINCIPAL COMPONENTS
# ==========================================

def transform_data(data, eigenvectors):

    result = []

    for row in data:

        new_row = []

        for vector in eigenvectors:

            total = 0

            for j in range(len(row)):
                total += row[j] * vector[j]

            new_row.append(total)

        result.append(new_row)

    return result


# ==========================================
# 14. APPLY PCA
# ==========================================

means = calculate_mean(data)

stds = calculate_std(data, means)

standardized_data = standardize_data(
    data, means, stds
)

covariance = covariance_matrix(standardized_data)

eigenvalues, eigenvectors = calculate_pca(
    covariance, 0.85
)

pca_result = transform_data(
    standardized_data, eigenvectors
)


# ==========================================
# 15. EXPLAINED VARIANCE
# ==========================================

total_variance = sum(eigenvalues)

# Note: use all original eigenvalues for the
# total variance calculation, not only retained ones.

all_eigenvalues = []
matrix = [row[:] for row in covariance]

for i in range(len(covariance)):

    value, vector = power_iteration(matrix)

    if value < 0:
        value = 0

    all_eigenvalues.append(value)

    matrix = deflate_matrix(matrix, value, vector)

total_variance = sum(all_eigenvalues)

explained_variance = [
    value / total_variance for value in eigenvalues
]

cumulative_variance = []
total = 0

for value in explained_variance:
    total += value
    cumulative_variance.append(total)


# ==========================================
# 16. CREATE PCA DATAFRAME
# ==========================================

pc_names = [
    "PC" + str(i + 1)
    for i in range(len(eigenvectors))
]

pca_df = pd.DataFrame(
    pca_result,
    columns=pc_names
)

print("\nPCA Transformed Data:")
print(pca_df.head())

print("\nOriginal Shape:", rows, cols)
print("Reduced Shape:", pca_df.shape)

print("\nExplained Variance:")
print(explained_variance)

print("\nCumulative Variance:")
print(cumulative_variance)


# ==========================================
# 17. PCA LOADINGS
# ==========================================

loadings = pd.DataFrame(
    transpose(eigenvectors),
    index=features,
    columns=pc_names
)

print("\nPCA Loadings:")
print(loadings)


# ==========================================
# 18. SAVE RESULTS
# ==========================================

with pd.ExcelWriter(OUTPUT_FOLDER / "Manual_PCA_Results.xlsx") as writer:

    pca_df.to_excel(
        writer,
        sheet_name="PCA_Scores",
        index=False
    )

    loadings.to_excel(
        writer,
        sheet_name="PCA_Loadings"
    )

    variance_df = pd.DataFrame({
        "Component": pc_names,
        "Eigenvalue": eigenvalues,
        "Explained_Variance": explained_variance,
        "Cumulative_Variance": cumulative_variance
    })

    variance_df.to_excel(
        writer,
        sheet_name="Explained_Variance",
        index=False
    )

print("\nManual PCA completed successfully!")
