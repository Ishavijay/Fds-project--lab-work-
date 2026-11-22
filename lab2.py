
import pandas as pd
import numpy as np #is a Python library used for numerical calculations, mathematical operations and working with arrays.
import matplotlib.pyplot as plt
import seaborn as sns #Python data visualization library used to create attractive and informative statistical graphs.

# 1. IMPORTING DATA
df = pd.read_excel("hotel_accounts_analysis_ready.xlsx")

# Create a new variable
df["Transaction_Amount_INR"] = (
    df["Debit_INR"] + df["Credit_INR"]
)

# 2. EXPLORING DATA

print("First five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# 3. DESCRIPTIVE STATISTICS

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nDescriptive Statistics:")
print(df[numeric_cols].describe()) #give the description about the numerical column

print("\nMedians:")
print(df[numeric_cols].median())


# 4. HANDLING MISSING VALUES


df_imputed = df.copy()

# Fill missing numerical values with median
for col in numeric_cols:
    df_imputed[col] = df_imputed[col].fillna(
        df_imputed[col].median()
    )

# Fill missing categorical values with mode
categorical_cols = df_imputed.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:
    if df_imputed[col].isnull().any():  #.any() :will check if any entry satify the condition
        df_imputed[col] = df_imputed[col].fillna(
            df_imputed[col].mode()[0]
        )

# 5. HANDLING OUTLIERS USING IQR 
df_final = df_imputed.copy()

# Exclude identifiers and derived diagnostic columns
outlier_cols = [
    "Debit_INR",
    "Credit_INR",
    "Transaction_Amount_INR",
    "GST_Amount_Original_INR",
    "GST_Rate"
]

for col in outlier_cols:

    Q1 = df_final[col].quantile(0.25)
    Q3 = df_final[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df_final[col] = df_final[col].clip(  #when we are trying to make the range between certain values
        lower=lower_limit,
        upper=upper_limit
    )

print("\nOutliers handled using IQR capping!")

#PLots

# 6. BAR CHART
payment_counts = df_final["Payment_Mode"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    payment_counts.index, # .index will give the labels
    payment_counts.values,#will provide the numerical values
    edgecolor="black"
)

plt.title("Payment Mode Distribution")
plt.xlabel("Payment Mode")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#SCATTER PLOT

plt.figure(figsize=(8, 5))

sns.stripplot(
    data=df_final,
    x="Department",
    y="Debit_INR",
    jitter=True
)

plt.title("Department vs Debit Amount")
plt.xlabel("Department")
plt.ylabel("Debit Amount (INR)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 8. BOX PLOT


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_final,
    x="Transaction_Type",
    y="Transaction_Amount_INR"
)

plt.title("Transaction Amount Distribution by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Amount (INR)")

plt.tight_layout()
plt.show()


# 9. HISTOGRAM

plt.figure(figsize=(8, 5))

plt.hist(
    df_final["Transaction_Amount_INR"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount (INR)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 10. LINE CHART


daily_transactions = (
    df_final.groupby("Transaction_Date")[
        "Transaction_Amount_INR"
    ].sum().sort_index()
)

plt.figure(figsize=(10, 5))

plt.plot(
    daily_transactions.index,
    daily_transactions.values,
    marker="o"
)

plt.title("Daily Transaction Amount Trend")
plt.xlabel("Transaction Date")
plt.ylabel("Total Transaction Amount (INR)")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


