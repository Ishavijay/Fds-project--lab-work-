"""Simple data analysis tool helpers for Unit VI."""


# Reads a CSV file into a pandas table.
def read_data_file(filename):
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Install pandas first: python -m pip install pandas"
        ) from error

    return pd.read_csv(filename)


# Marks a table column as categories, such as small, medium, or large.
def make_category(data, column, categories=None):
    result = data.copy()
    result[column] = result[column].astype("category")

    if categories is not None:
        result[column] = result[column].cat.set_categories(categories)

    return result


# Makes a NumPy matrix from selected columns in a table.
def make_matrix(data, columns=None):
    if columns is not None:
        data = data[columns]

    return data.to_numpy()


# Converts one table column into a regular Python list.
def column_as_list(data, column):
    return data[column].tolist()


# Shows common statistics for one column, such as its average and spread.
def summarize_column(data, column):
    return data[column].describe()


# Counts how often each value appears in a column.
def count_values(data, column):
    return data[column].value_counts(dropna=False)

