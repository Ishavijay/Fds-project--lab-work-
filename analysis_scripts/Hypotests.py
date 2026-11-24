from pathlib import Path

import pandas as pd
# 1. READ DATA

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "hotel_accounts_raw_100.xlsx"
df = pd.read_excel(DATA_FILE)
#significance level used for all type of test is 0.05
# Create transaction amount column
amount = []
#we are trying to create a new column transaction amount
for i in range(len(df)):
    amount.append( #append the credit and debit value into amount list
        float(df.loc[i, "Debit (INR)"]) +   # gets value from ith row of debit  and converts into decimal 
        float(df.loc[i, "Credit (INR)"])    #get value from ith row of credit 
    )

df["Transaction_Amount"] = amount

# 2. MANUAL STATISTICAL FUNCTIONS

def mean(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)


def variance(data):
    avg = mean(data)
    total = 0

    for x in data:
        total += (x - avg) ** 2

    return total / (len(data) - 1)


def standard_deviation(data):
    return variance(data) ** 0.5

# 3. PAIRED SAMPLE T-TEST
# Debit vs Credit difference is 0 or not

debit = []
credit = []
difference = []

for i in range(len(df)):

    d = float(df.loc[i, "Debit (INR)"])
    c = float(df.loc[i, "Credit (INR)"])

    debit.append(d)
    credit.append(c)
    difference.append(d - c)

n = len(difference)

mean_diff = mean(difference)
sd_diff = standard_deviation(difference)

t_paired = mean_diff / (sd_diff / (n ** 0.5))

df_paired = n - 1

print("\n========== PAIRED SAMPLE T-TEST ==========")

print("Mean Debit:", mean(debit))
print("Mean Credit:", mean(credit))
print("Mean Difference:", mean_diff)
print("SD of Differences:", sd_diff)
print("T Statistic:", t_paired)
print("Degrees of Freedom:", df_paired)

critical_t_paired = 1.984

if abs(t_paired) > critical_t_paired:
    print("Reject H0")
else:
    print("Fail to Reject H0")



# 4. INDEPENDENT SAMPLE T-TEST
# Revenue Receipt vs Operating Expense
# Welch's T-Test

#we check if Mean Revenue Transaction Amount = Mean Expense Transaction Amount.
revenue = []
expense = []

for i in range(len(df)):

    value = float(df.loc[i, "Transaction_Amount"])

    if df.loc[i, "Transaction Type"] == "Revenue Receipt":
        revenue.append(value)

    elif df.loc[i, "Transaction Type"] == "Operating Expense":
        expense.append(value)

n1 = len(revenue)
n2 = len(expense)

mean1 = mean(revenue)
mean2 = mean(expense)

var1 = variance(revenue)
var2 = variance(expense)

t_independent = (mean1 - mean2) / (
    (var1/n1 + var2/n2) ** 0.5
)

# Welch degrees of freedom
df_independent = (
    (var1/n1 + var2/n2) ** 2
) / (
    ((var1/n1)**2 / (n1-1)) +
    ((var2/n2)**2 / (n2-1))
)

print("\n========== INDEPENDENT SAMPLE T-TEST ==========")

print("Revenue Mean:", mean1)
print("Expense Mean:", mean2)
print("Revenue Sample Size:", n1)
print("Expense Sample Size:", n2)
print("T Statistic:", t_independent)
print("Welch Degrees of Freedom:", df_independent)

print("Use the t-table with the calculated degrees of freedom.")
print("Compare against the two-tailed critical value at alpha = 0.05.")


# 5. ONE-SAMPLE T-TEST
# Transaction Amount vs Rs. 5000

#We want to determine whether the average transaction amount is significantly different from ₹5,000.
sample = df["Transaction_Amount"].tolist()

n = len(sample)

sample_mean = mean(sample)
sample_sd = standard_deviation(sample)

hypothesized_mean = 5000

t_one_sample = (
    (sample_mean - hypothesized_mean) /
    (sample_sd / (n ** 0.5))
)

df_one_sample = n - 1

print("\n========== ONE-SAMPLE T-TEST ==========")

print("Sample Mean:", sample_mean)
print("Sample SD:", sample_sd)
print("Hypothesized Mean:", hypothesized_mean)
print("T Statistic:", t_one_sample)
print("Degrees of Freedom:", df_one_sample)

critical_t_one_sample = 1.984

if abs(t_one_sample) > critical_t_one_sample:
    print("Reject H0")
else:
    print("Fail to Reject H0")



# 6. ONE-WAY ANOVA
# Transaction Amount across Departments
#we want to determine whether average transaction amounts differ across the eight hotel departments.
departments = []

for i in range(len(df)):

    department = df.loc[i, "Department"]

    if department not in departments:
        departments.append(department)

groups = []   #creates an empty list to store all department groups.

for department in departments:

    group = []   #creates a separate empty list for each department.

    for i in range(len(df)):

        if df.loc[i, "Department"] == department:
            group.append(                   #adds the transaction amount to that department's list.
                float(df.loc[i, "Transaction_Amount"])      #stores the completed department list inside groups
            )

    groups.append(group)


# Grand mean
all_values = []

for group in groups:
    for value in group:
        all_values.append(value)

grand_mean = mean(all_values)

# Between-group sum of squares
SS_between = 0

for group in groups:

    group_mean = mean(group)

    SS_between += (
        len(group) * (group_mean - grand_mean) ** 2
    )

# Within-group sum of squares
SS_within = 0

for group in groups:

    group_mean = mean(group)

    for value in group:
        SS_within += (value - group_mean) ** 2


k = len(groups)
N = len(all_values)

df_between = k - 1
df_within = N - k

MS_between = SS_between / df_between
MS_within = SS_within / df_within

F_one_way = MS_between / MS_within

print("\n========== ONE-WAY ANOVA ==========")

print("Departments:", departments)
print("Grand Mean:", grand_mean)
print("SS Between:", SS_between)
print("SS Within:", SS_within)
print("MS Between:", MS_between)
print("MS Within:", MS_within)
print("F Statistic:", F_one_way)
print("Degrees of Freedom:", df_between, df_within)

critical_F_one_way = 2.11

if F_one_way > critical_F_one_way:
    print("Reject H0")
else:
    print("Fail to Reject H0")


# 7. TWO-WAY ANOVA
# Department and Transaction Type
# Main Effects and Interaction
#We want to study how two categorical variables influence transaction amounts:
#1. Department
#2. Transaction Type
#We investigate three things:

#A. Department effect

#Does the average transaction amount vary across departments?

#B. Transaction Type effect

#Does the average transaction amount differ between Revenue Receipt and Operating Expense?

#C. Interaction effect

#Does the difference between Revenue Receipt and Operating Expense depend on the department?
departments = sorted(df["Department"].unique().tolist())
types = sorted(df["Transaction Type"].unique().tolist())

# Treatment coding:
# First department and first type are reference categories.

base_department = departments[0]
base_type = types[0]

X_full = []
Y = []

for i in range(len(df)):

    d = df.loc[i, "Department"]
    t = df.loc[i, "Transaction Type"]

    row = [1.0]

    # Department dummy variables
    for department in departments[1:]:
        row.append(1.0 if d == department else 0.0)

    # Transaction type dummy variables
    for transaction_type in types[1:]:
        row.append(1.0 if t == transaction_type else 0.0)

    # Interaction variables
    for department in departments[1:]:
        for transaction_type in types[1:]:
            row.append(
                1.0 if (
                    d == department and
                    t == transaction_type
                ) else 0.0
            )

    X_full.append(row)
    Y.append(float(df.loc[i, "Transaction_Amount"]))


# MANUAL GAUSS-JORDAN ELIMINATION

def solve_linear_system(A, b):

    n = len(b)
    matrix = []

    for i in range(n):
        row = []

        for j in range(n):
            row.append(float(A[i][j]))

        row.append(float(b[i]))
        matrix.append(row)

    for i in range(n):

        # Partial pivoting
        pivot_row = i

        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[pivot_row][i]):
                pivot_row = k

        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        pivot = matrix[i][i]

        if abs(pivot) < 1e-10:
            raise ValueError(
                "Singular matrix: some category combinations may be missing."
            )

        for j in range(i, n + 1):
            matrix[i][j] /= pivot

        for k in range(n):

            if k != i:

                factor = matrix[k][i]

                for j in range(i, n + 1):
                    matrix[k][j] -= factor * matrix[i][j]

    solution = []

    for i in range(n):
        solution.append(matrix[i][n])

    return solution


# ------------------------------------------
# CALCULATE RESIDUAL SUM OF SQUARES
# ------------------------------------------

def calculate_SSE(X, Y):

    rows = len(X)
    cols = len(X[0])

    # Calculate X transpose X
    A = []

    for j in range(cols):

        row = []

        for k in range(cols):

            total = 0

            for i in range(rows):
                total += X[i][j] * X[i][k]

            row.append(total)

        A.append(row)

    # Calculate X transpose Y
    b = []

    for j in range(cols):

        total = 0

        for i in range(rows):
            total += X[i][j] * Y[i]

        b.append(total)

    beta = solve_linear_system(A, b)

    # Calculate SSE
    SSE = 0

    for i in range(rows):

        predicted = 0

        for j in range(cols):
            predicted += X[i][j] * beta[j]

        error = Y[i] - predicted
        SSE += error ** 2

    return SSE


# Column positions
department_columns = list(range(1, len(departments)))

type_start = len(departments)
type_columns = list(
    range(type_start, type_start + len(types) - 1)
)

interaction_start = type_start + len(types) - 1
interaction_columns = list(
    range(interaction_start, len(X_full[0]))
)


# Full model
SSE_full = calculate_SSE(X_full, Y)

# Additive model: no interaction
additive_columns = (
    [0] +
    department_columns +
    type_columns
)

X_additive = [
    [row[j] for j in additive_columns]
    for row in X_full
]

SSE_additive = calculate_SSE(X_additive, Y)


# Model without Department
no_department_columns = [0] + type_columns + interaction_columns

X_no_department = [
    [row[j] for j in no_department_columns]
    for row in X_full
]

SSE_no_department = calculate_SSE(X_no_department, Y)


# Model without Transaction Type
no_type_columns = [0] + department_columns + interaction_columns

X_no_type = [
    [row[j] for j in no_type_columns]
    for row in X_full
]

SSE_no_type = calculate_SSE(X_no_type, Y)


# Degrees of freedom
N = len(Y)
full_parameters = len(X_full[0])
error_df = N - full_parameters

if error_df <= 0:
    raise ValueError("Insufficient residual degrees of freedom.")

MSE = SSE_full / error_df

department_df = len(department_columns)
type_df = len(type_columns)
interaction_df = len(interaction_columns)


# Partial F statistics
F_department = (
    ((SSE_no_department - SSE_full) / department_df) / MSE
)

F_type = (
    ((SSE_no_type - SSE_full) / type_df) / MSE
)

F_interaction = (
    ((SSE_additive - SSE_full) / interaction_df) / MSE
)

print("\n========== TWO-WAY ANOVA ==========")

print("Full Model SSE:", SSE_full)
print("Additive Model SSE:", SSE_additive)
print("Error Degrees of Freedom:", error_df)
print("MSE:", MSE)

print("\nDepartment F:", F_department)
print("Department df:", department_df, error_df)

print("\nTransaction Type F:", F_type)
print("Type df:", type_df, error_df)

print("\nInteraction F:", F_interaction)
print("Interaction df:", interaction_df, error_df)

print(
    "\nNote: These are treatment-coded partial tests. "
    "Main-effect tests are conditional on the chosen reference coding."
)


# ==========================================
# 8. ONE-SAMPLE Z-TEST
# Transaction Amount vs Rs. 5000
# ==========================================

# Large-sample approximation using sample SD.
# Classical Z-test normally requires known population SD.

n = len(sample)

z_stat = (
    (sample_mean - hypothesized_mean) /
    (sample_sd / (n ** 0.5))
)

print("\n========== ONE-SAMPLE Z-TEST ==========")

print("Sample Mean:", sample_mean)
print("Sample SD:", sample_sd)
print("Hypothesized Mean:", hypothesized_mean)
print("Z Statistic:", z_stat)

critical_z = 1.96

if abs(z_stat) > critical_z:
    print("Reject H0")
else:
    print("Fail to Reject H0")



# 9. FINAL SUMMARY


print("\n========== ALL TESTS COMPLETED ==========")

print("1. Paired Sample t-test")
print("2. Independent Sample t-test")
print("3. One-Sample t-test")
print("4. One-Way ANOVA")
print("5. Two-Way ANOVA")
print("6. One-Sample Z-test")
