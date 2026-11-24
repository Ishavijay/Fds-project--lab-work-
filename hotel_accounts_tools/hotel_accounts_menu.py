"""Run the project functions from a simple menu."""

from pathlib import Path
import sys


# The workbook stays in the project folder; results go inside this tools folder.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_FOLDER / "hotel_accounts_analysis_ready.xlsx"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"
sys.path.insert(0, str(PROJECT_FOLDER))

from fds_unit_functions.unit_1_intro_data_science import (
    analytics_types,
    crisp_dm_steps,
    semma_steps,
)
from fds_unit_functions.unit_2_probability_statistics import (
    average,
    conditional_probability,
    correlation,
    normal_probability,
)
from fds_unit_functions.unit_3_data_sources_visualization import (
    describe_data,
    fill_missing,
    load_csv,
    show_histogram,
    show_scatter_plot,
)
from fds_unit_functions.unit_4_feature_engineering_optimization import (
    fit_linear_model,
    gradient_descent,
    make_category_columns,
    scale_numbers,
)
from fds_unit_functions.unit_5_machine_learning import (
    check_accuracy,
    find_frequent_items,
    make_clusters,
    make_predictions,
    train_classifier,
)
from fds_unit_functions.unit_6_data_analysis_tools import (
    column_as_list,
    count_values,
    make_category,
    make_matrix,
    read_data_file,
    summarize_column,
)


# Read the Analysis_Ready sheet from the Excel workbook.
def load_dataset():
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Install pandas and openpyxl: python -m pip install pandas openpyxl"
        ) from error

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    return pd.read_excel(DATA_FILE, sheet_name="Analysis_Ready")


# Show the three planning helpers from Unit I.
def apply_unit_1(data):
    print("\nAnalytics types:")
    for name, question in analytics_types().items():
        print(f"{name.title()}: {question}")

    print("\nCRISP-DM steps:")
    for step in crisp_dm_steps():
        print(f"- {step}")

    print("\nSEMMA steps:")
    print(", ".join(semma_steps()))

    print(f"\nThese steps can guide work on {len(data)} hotel transactions.")


# Use Unit II functions with transaction amounts, dates and GST rates.
def apply_unit_2(data):
    amounts = data["Transaction_Amount_INR"].dropna().tolist()
    amount_average = average(amounts)

    taxable = data["GST_Rate"] > 0
    revenue = data["Transaction_Type"] == "Revenue Receipt"
    probability_taxable = taxable.mean()
    probability_both = (taxable & revenue).mean()
    revenue_given_taxable = conditional_probability(
        probability_both, probability_taxable
    )

    day_and_amount = data[["Day_Index", "Transaction_Amount_INR"]].dropna()
    day_amount_correlation = correlation(
        day_and_amount["Day_Index"].tolist(),
        day_and_amount["Transaction_Amount_INR"].tolist(),
    )
    amount_std = data["Transaction_Amount_INR"].std()
    top_quartile = data["Transaction_Amount_INR"].quantile(0.75)
    top_quartile_probability = normal_probability(
        top_quartile, amount_average, amount_std
    )

    print(f"\nAverage transaction amount: INR {amount_average:,.2f}")
    print(
        "Chance of revenue when GST is charged: "
        f"{revenue_given_taxable:.1%}"
    )
    print(f"Correlation between day and amount: {day_amount_correlation:.2f}")
    print(
        "Normal-curve estimate below the 75th percentile: "
        f"{top_quartile_probability:.1%}"
    )


# Load the workbook with Unit III tools, summarize, fill gaps, and show two plots.
def apply_unit_3(data):
    unit_data = load_csv(DATA_FILE)
    numeric_columns = [
        "Transaction_Amount_INR",
        "GST_Rate",
        "Day_Index",
    ]
    numeric_data = unit_data[numeric_columns]

    print("\nNumeric data summary:")
    print(describe_data(numeric_data).round(2))

    filled_data = fill_missing(numeric_data, numeric_data.median())
    print(f"\nMissing numeric values after filling: {filled_data.isna().sum().sum()}")

    print("\nShowing transaction amount histogram and day scatter plot.")
    show_histogram(unit_data, "Transaction_Amount_INR")
    show_scatter_plot(unit_data, "Day_Index", "Transaction_Amount_INR")


# Use Unit IV helpers to encode categories, scale numbers and fit simple models.
def apply_unit_4(data):
    encoded_data = make_category_columns(data, "Transaction_Type")
    print(f"\nColumns after category encoding: {len(encoded_data.columns)}")

    number_columns = ["Transaction_Amount_INR", "GST_Rate", "Day_Index"]
    scaled_data = scale_numbers(data[number_columns])
    print(f"Scaled numeric data shape: {scaled_data.shape}")

    model = fit_linear_model(
        data[["Day_Index"]], data["Transaction_Amount_INR"]
    )
    print(f"Linear model slope by day: INR {model.coef_[0]:,.2f}")

    target_average = data["Transaction_Amount_INR"].mean()
    estimate = gradient_descent(
        lambda value: 2 * (value - target_average),
        start=0,
        step_size=0.1,
        steps=100,
    )
    print(f"Gradient descent estimate of the average: INR {estimate:,.2f}")


# Run the Unit V classifier, accuracy check, clustering and frequent-item search.
def apply_unit_5(data):
    from numpy import unique
    from sklearn.model_selection import train_test_split

    feature_columns = [
        "Transaction_Amount_INR",
        "GST_Rate",
        "GST_Amount_Recalculated_INR",
        "Day_Index",
    ]
    features = data[feature_columns]
    answers = data["Transaction_Type"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        answers,
        test_size=0.2,
        random_state=42,
        stratify=answers,
    )

    model = train_classifier(x_train, y_train)
    predictions = make_predictions(model, x_test)
    accuracy = check_accuracy(model, x_test, y_test)
    print(f"\nClassifier test accuracy: {accuracy:.1%}")
    print(f"First five predictions: {list(predictions[:5])}")
    print("This is a practice result from a small dataset.")

    cluster_columns = ["Transaction_Amount_INR", "Day_Index"]
    scaled_data = scale_numbers(data[cluster_columns])
    cluster_labels = make_clusters(scaled_data, number_of_groups=3)
    print("\nRecords in each cluster:")
    cluster_numbers, cluster_counts = unique(
        cluster_labels, return_counts=True
    )
    print(dict(zip(cluster_numbers, cluster_counts)))

    baskets = data[
        ["Department", "Account_Head", "Payment_Mode"]
    ].astype(str).values.tolist()
    common_items = find_frequent_items(baskets, minimum_fraction=0.2)
    print("\nCommon item or category pairs:")
    for items, support in sorted(
        common_items.items(), key=lambda result: result[1], reverse=True
    )[:10]:
        print(f"{items}: {support:.0%}")


# Use Unit VI helpers for categories, lists, matrices and distributions.
def apply_unit_6(data):
    unit_data = read_data_file(DATA_FILE)
    category_data = make_category(unit_data, "Transaction_Type")
    print(f"\nTransaction type data type: {category_data['Transaction_Type'].dtype}")

    columns = ["Transaction_Amount_INR", "GST_Rate", "Day_Index"]
    matrix = make_matrix(unit_data, columns)
    print(f"Numeric matrix shape: {matrix.shape}")

    transaction_types = column_as_list(unit_data, "Transaction_Type")
    print(f"First five transaction types: {transaction_types[:5]}")

    print("\nTransaction amount distribution:")
    print(summarize_column(unit_data, "Transaction_Amount_INR"))

    print("\nCount by transaction type:")
    print(count_values(unit_data, "Transaction_Type"))

    print(f"\nThe menu loaded {len(data)} transactions for this example.")


# Preview the dataset.
def show_preview(data):
    print("\nFirst five rows:")
    print(data.head())
    print(f"\nRows: {len(data)} | Columns: {len(data.columns)}")


# Summarize revenue and expense totals and averages.
def show_summary(data):
    summary = data.groupby("Transaction_Type")[
        "Transaction_Amount_INR"
    ].agg(["count", "mean", "sum"])
    summary.columns = ["Records", "Average INR", "Total INR"]
    print("\nRevenue and expense summary:")
    print(summary.round(2))


# Count missing cells and duplicate rows.
def check_data_quality(data):
    print(f"\nMissing cells: {int(data.isna().sum().sum())}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")


# Remove duplicate rows and save a separate workbook copy.
def clean_and_save_copy(data):
    cleaned_data = data.drop_duplicates().copy()
    removed_rows = len(data) - len(cleaned_data)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / "hotel_accounts_cleaned.xlsx"
    cleaned_data.to_excel(output_file, index=False)

    print(f"\nRemoved duplicate rows: {removed_rows}")
    print(f"Saved cleaned copy: {output_file}")


# Create and save a chart of average amount by transaction type.
def save_amount_chart(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Install matplotlib: python -m pip install matplotlib"
        ) from error

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / "average_amount_by_type.png"
    averages = data.groupby("Transaction_Type")[
        "Transaction_Amount_INR"
    ].mean()

    averages.plot(kind="bar", color=["#2D6FA5", "#E58A3A"])
    plt.title("Average Transaction Amount by Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Average Amount (INR)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.show()
    plt.close()
    print(f"Chart saved: {output_file}")


# Ask what to run and use match-case to select the action.
def main():
    try:
        data = load_dataset()
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"Could not load the workbook: {error}")
        return

    while True:
        print("\nHotel Accounts Project")
        print("1. Run Unit I functions")
        print("2. Run Unit II functions")
        print("3. Run Unit III functions and plots")
        print("4. Run Unit IV functions")
        print("5. Run Unit V functions")
        print("6. Run Unit VI functions")
        print("7. Preview the dataset")
        print("8. Summarize revenue and expenses")
        print("9. Check data quality")
        print("10. Save a copy without duplicate rows")
        print("11. Save an average amount chart")
        print("0. Exit")

        choice = input("Choose an option: ").strip()

        try:
            match choice:
                case "1":
                    apply_unit_1(data)
                case "2":
                    apply_unit_2(data)
                case "3":
                    apply_unit_3(data)
                case "4":
                    apply_unit_4(data)
                case "5":
                    apply_unit_5(data)
                case "6":
                    apply_unit_6(data)
                case "7":
                    show_preview(data)
                case "8":
                    show_summary(data)
                case "9":
                    check_data_quality(data)
                case "10":
                    clean_and_save_copy(data)
                case "11":
                    save_amount_chart(data)
                case "0":
                    print("Goodbye.")
                    break
                case _:
                    print("Choose one of the listed options.")
        except (ImportError, KeyError, ValueError) as error:
            print(f"Could not run that option: {error}")


if __name__ == "__main__":
    main()
