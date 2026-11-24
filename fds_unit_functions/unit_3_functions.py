# Unit III: loading, describing, cleaning, and visualizing data.
# CSV and Excel files are loaded as pandas DataFrames; chart functions display plots.

import csv
import math
from pathlib import Path


def load_csv(filename):
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


def describe_data(data):
    # Return summary statistics for every column in the table.
    import pandas as pd

    summary = {}
    for column in data.columns:
        values = [value for value in data[column] if not _is_missing(value)]
        if not values:
            summary[column] = {"count": 0}
            continue

        if all(_is_number(value) for value in values):
            numbers = [float(value) for value in values]
            summary[column] = {
                "count": len(numbers),
                "mean": _mean(numbers),
                "std": _standard_deviation(numbers),
                "min": min(numbers),
                "25%": _quantile(numbers, 0.25),
                "50%": _quantile(numbers, 0.50),
                "75%": _quantile(numbers, 0.75),
                "max": max(numbers),
            }
        else:
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            top_value = max(counts, key=counts.get)
            summary[column] = {
                "count": len(values),
                "unique": len(counts),
                "top": top_value,
                "freq": counts[top_value],
            }

    return pd.DataFrame(summary)


def fill_missing(data, value=0):
    # Return a copy of the table with missing cells replaced by the chosen value.
    result = data.copy()
    for row_index in range(len(result)):
        for column_index in range(len(result.columns)):
            if _is_missing(result.iat[row_index, column_index]):
                result.iat[row_index, column_index] = value

    return result


def show_histogram(data, column):
    # Display a histogram showing the values in one data column.
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Install matplotlib first: python -m pip install matplotlib"
        ) from error

    values = [value for value in data[column] if not _is_missing(value)]
    if not values:
        raise ValueError("The selected column has no values to plot.")

    minimum = min(values)
    maximum = max(values)
    bin_count = min(10, max(1, int(len(values) ** 0.5)))
    bin_width = (maximum - minimum) / bin_count if maximum != minimum else 1
    counts = [0] * bin_count
    centers = []
    for value in values:
        bin_index = min(int((value - minimum) / bin_width), bin_count - 1)
        counts[bin_index] += 1

    for bin_index in range(bin_count):
        centers.append(minimum + (bin_index + 0.5) * bin_width)

    plt.bar(centers, counts, width=bin_width * 0.9, edgecolor="black")
    plt.title("Values in " + column)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()


def show_scatter_plot(data, x_column, y_column):
    # Display a scatter plot comparing two columns in the table.
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Install matplotlib first: python -m pip install matplotlib"
        ) from error

    points = []
    for x_value, y_value in zip(data[x_column], data[y_column]):
        if not _is_missing(x_value) and not _is_missing(y_value):
            points.append((x_value, y_value))
    if not points:
        raise ValueError("The selected columns have no matching values to plot.")

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    plt.scatter(x_values, y_values)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


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
    # Check for common empty values without calling a DataFrame helper.
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


def _standard_deviation(values):
    # Calculate sample standard deviation using squared differences.
    if len(values) < 2:
        return float("nan")
    mean = _mean(values)
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

