
from pathlib import Path

import pandas as pd
import math # a built in python module 

# Read the Excel file
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "hotel_accounts_analysis_ready.xlsx"
df = pd.read_excel(DATA_FILE,
                   sheet_name="Analysis_Ready")



# MEAN

def mean_of_column(df, column_name): # df and column_name are parameters

    total = 0
    count = 0

    # Remove missing values
    for value in df[column_name].dropna():

        total += value
        count += 1

    if count == 0:
        return None

    return total / count


# MEDIAN

def median_of_column(df, column_name):

    # Remove null values and sort
    values = sorted(df[column_name].dropna())

    n = len(values)

    if n == 0:
        return None

    middle = n // 2

    if n % 2 == 0:

        return (
            values[middle - 1] +
            values[middle]
        ) / 2

    else:

        return values[middle]


# POPULATION VARIANCE

def variance_of_column(df, column_name):

    mean = mean_of_column(df, column_name)

    total = 0
    count = 0

    for value in df[column_name].dropna():

        total += (value - mean) ** 2
        count += 1

    if count == 0:
        return None

    # Population variance
    return total / count

#method : Is a function associated with a particular class or object 
#function:independent block of code that performs a particular task
#Attributes are variables that belong to an object and store information about that object.

# POPULATION STANDARD DEVIATION


def standard_deviation_of_column(df, column_name):

    variance = variance_of_column(df, column_name)

    if variance is None:
        return None

    return math.sqrt(variance)


# SELECT COLUMN

amount_column = "Net_Amount_Recalculated_INR"



# PRINT RESULTS

print("Mean:")
print(mean_of_column(df, amount_column)) # df and amount_column are arguments

print("\nMedian:")
print(median_of_column(df, amount_column))

print("\nPopulation Variance:")
print(variance_of_column(df, amount_column))

print("\nPopulation Standard Deviation:")
print(standard_deviation_of_column(df, amount_column))
