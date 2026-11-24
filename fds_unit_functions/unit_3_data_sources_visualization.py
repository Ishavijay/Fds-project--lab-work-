"""Simple data and chart helpers for Unit III."""


# Reads a CSV or Excel file into a pandas table called a DataFrame.
def load_csv(filename):
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Install pandas and openpyxl: python -m pip install pandas openpyxl"
        ) from error
    if str(filename).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(filename, sheet_name="Analysis_Ready")

    return pd.read_csv(filename)


# Shows each column's type, missing values, and basic statistics.
def describe_data(data):
    return data.describe(include="all")


# Replaces missing cells with the value you choose (zero by default).
def fill_missing(data, value=0):
    return data.fillna(value)


# Draws a histogram so you can see the spread of values in one column.
def show_histogram(data, column):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Install matplotlib first: python -m pip install matplotlib"
        ) from error
    data[column].dropna().hist(edgecolor="black")
    plt.title("Values in " + column)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()


# Draws a scatter plot to show how two columns are related.
def show_scatter_plot(data, x_column, y_column):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Install matplotlib first: python -m pip install matplotlib"
        ) from error
    plt.scatter(data[x_column], data[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

