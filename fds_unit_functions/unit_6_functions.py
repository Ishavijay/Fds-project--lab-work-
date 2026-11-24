# Unit VI: common operations for working with tabular data.
# These helpers read files, work with categories, and summarize table columns.

import csv
import math
from pathlib import Path


def read_data_file(filename):
    # Load a CSV or Excel file and return its data as a DataFrame.
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Install pandas and openpyxl: python -m pip install pandas openpyxl"
        ) from error

    if str(filename).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(filename, sheet_name="Analysis_Ready")

    with Path(filename).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = []
        for source_row in reader:
            row = {}
            for column, value in source_row.items():
                row[column] = _convert_cell(value)
            rows.append(row)

    return pd.DataFrame(rows)


def make_category(data, column, categories=None):
    # Return a copy with the chosen column stored as a category.
    import pandas as pd

    result = data.copy()
    values = list(result[column])
    if categories is None:
        categories = []
        for value in values:
            if not _is_missing(value) and value not in categories:
                categories.append(value)

    result[column] = pd.Categorical(values, categories=categories)

    return result


def make_matrix(data, columns=None):
    # Return selected table values as a NumPy array.
    import numpy as np

    if columns is not None:
        data = data[columns]
    rows = []
    for row_index in range(len(data)):
        row = []
        for column_index in range(len(data.columns)):
            row.append(data.iat[row_index, column_index])
        rows.append(row)

    return np.array(rows)


def column_as_list(data, column):
    # Return the values in the chosen column as a regular Python list.
    values = []
    for value in data[column]:
        values.append(value)

    return values


def summarize_column(data, column):
    # Return summary statistics for one column.
    import pandas as pd

    values = [value for value in data[column] if not _is_missing(value)]
    if not values:
        return pd.Series({"count": 0})

    if all(_is_number(value) for value in values):
        numbers = [float(value) for value in values]
        mean = _mean(numbers)
        return pd.Series(
            {
                "count": len(numbers),
                "mean": mean,
                "std": _standard_deviation(numbers, mean),
                "min": min(numbers),
                "25%": _quantile(numbers, 0.25),
                "50%": _quantile(numbers, 0.50),
                "75%": _quantile(numbers, 0.75),
                "max": max(numbers),
            }
        )

    counts = _count_values(values)
    most_common = max(counts, key=counts.get)
    return pd.Series(
        {
            "count": len(values),
            "unique": len(counts),
            "top": most_common,
            "freq": counts[most_common],
        }
    )


def count_values(data, column):
    # Return the count of each value in a column, including missing values.
    import pandas as pd

    counts = {}
    for value in data[column]:
        key = "<missing>" if _is_missing(value) else value
        counts[key] = counts.get(key, 0) + 1

    return pd.Series(counts)


def _convert_cell(value):
    # Convert one CSV value to a number when possible.
    if value is None or value == "":
        return float("nan")
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _is_missing(value):
    # Check for common empty values without using a table helper.
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_number(value):
    # Return whether a value is a real number and not a Boolean.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values):
    # Calculate the arithmetic mean using a running total.
    total = 0
    for value in values:
        total += value

    return total / len(values)


def _standard_deviation(values, mean):
    # Calculate sample standard deviation using squared differences.
    if len(values) < 2:
        return float("nan")
    squared_total = 0
    for value in values:
        squared_total += (value - mean) ** 2

    return math.sqrt(squared_total / (len(values) - 1))


def _quantile(values, fraction):
    # Find a linearly interpolated quantile after sorting the values.
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def _count_values(values):
    # Count each value by adding one to its running frequency.
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts

