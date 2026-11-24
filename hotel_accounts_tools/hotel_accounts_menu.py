"""Choose a data or analysis function for the hotel accounts workbook."""

from pathlib import Path
import sys


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_FOLDER / "data" / "hotel_accounts_analysis_ready.xlsx"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "hotel_accounts_tools"
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


# Open the analysis-ready sheet in the hotel accounts workbook.
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


# Print a short preview of the workbook.
def show_preview(data):
    print("\nFirst five rows:")
    print(data.head())
    print(f"\nRows: {len(data)} | Columns: {len(data.columns)}")


# Show record counts, average amount and total amount by transaction type.
def show_summary(data):
    summary = data.groupby("Transaction_Type")[
        "Transaction_Amount_INR"
    ].agg(["count", "mean", "sum"])
    summary.columns = ["Records", "Average INR", "Total INR"]
    print("\nRevenue and expense summary:")
    print(summary.round(2))


# Count missing cells and duplicate rows without changing the data.
def check_data_quality(data):
    print(f"\nMissing cells: {int(data.isna().sum().sum())}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")


# Save a copy without exact duplicate rows. The original workbook stays as is.
def clean_and_save_copy(data):
    cleaned_data = data.drop_duplicates().copy()
    removed_rows = len(data) - len(cleaned_data)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / "hotel_accounts_cleaned.xlsx"
    cleaned_data.to_excel(output_file, index=False)

    print(f"\nRemoved duplicate rows: {removed_rows}")
    print(f"Saved cleaned copy: {output_file}")


# Make and save a bar chart of average amounts by transaction type.
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


# Prepare the columns used by the classifier examples.
def classifier_columns(data):
    feature_names = [
        "Transaction_Amount_INR",
        "GST_Rate",
        "GST_Amount_Recalculated_INR",
        "Day_Index",
    ]
    return data[feature_names], data["Transaction_Type"]


# Show the available function names before asking for a choice.
def show_menu():
    print("\nHotel accounts functions")
    print(" 1. analytics_types")
    print(" 2. crisp_dm_steps")
    print(" 3. semma_steps")
    print(" 4. average")
    print(" 5. conditional_probability")
    print(" 6. correlation")
    print(" 7. normal_probability")
    print(" 8. load_csv")
    print(" 9. describe_data")
    print("10. fill_missing")
    print("11. show_histogram")
    print("12. show_scatter_plot")
    print("13. make_category_columns")
    print("14. scale_numbers")
    print("15. fit_linear_model")
    print("16. gradient_descent")
    print("17. train_classifier")
    print("18. make_predictions")
    print("19. check_accuracy")
    print("20. make_clusters")
    print("21. find_frequent_items")
    print("22. read_data_file")
    print("23. make_category")
    print("24. make_matrix")
    print("25. column_as_list")
    print("26. summarize_column")
    print("27. count_values")
    print("28. show_preview")
    print("29. show_summary")
    print("30. check_data_quality")
    print("31. clean_and_save_copy")
    print("32. save_amount_chart")
    print("33. classifier_columns")
    print(" 0. Exit")


# Ask which function to run and call it in the matching case.
def main():
    try:
        data = load_dataset()
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"Could not load the workbook: {error}")
        return

    while True:
        show_menu()
        choice = input("Choose a function: ").strip().lower()

        try:
            match choice:
                case "1" | "analytics_types":
                    print(analytics_types())
                case "2" | "crisp_dm_steps":
                    print("\n".join(crisp_dm_steps()))
                case "3" | "semma_steps":
                    print(", ".join(semma_steps()))
                case "4" | "average":
                    print(average(data["Transaction_Amount_INR"].dropna().tolist()))
                case "5" | "conditional_probability":
                    taxable = data["GST_Rate"] > 0
                    revenue = data["Transaction_Type"] == "Revenue Receipt"
                    chance_of_both = (taxable & revenue).mean()
                    chance_of_b = taxable.mean()
                    print(conditional_probability(chance_of_both, chance_of_b))
                case "6" | "correlation":
                    paired = data[["Day_Index", "Transaction_Amount_INR"]].dropna()
                    print(correlation(
                        paired["Day_Index"].tolist(),
                        paired["Transaction_Amount_INR"].tolist(),
                    ))
                case "7" | "normal_probability":
                    amounts = data["Transaction_Amount_INR"]
                    print(normal_probability(
                        amounts.quantile(0.75), amounts.mean(), amounts.std()
                    ))
                case "8" | "load_csv":
                    loaded_data = load_csv(DATA_FILE)
                    print(loaded_data.head())
                    print(f"Rows and columns: {loaded_data.shape}")
                case "9" | "describe_data":
                    columns = ["Transaction_Amount_INR", "GST_Rate", "Day_Index"]
                    print(describe_data(data[columns]).round(2))
                case "10" | "fill_missing":
                    columns = ["Transaction_Amount_INR", "GST_Rate", "Day_Index"]
                    numbers = data[columns]
                    cleaned_numbers = fill_missing(numbers, numbers.median())
                    print(cleaned_numbers.head())
                case "11" | "show_histogram":
                    show_histogram(data, "Transaction_Amount_INR")
                case "12" | "show_scatter_plot":
                    show_scatter_plot(data, "Day_Index", "Transaction_Amount_INR")
                case "13" | "make_category_columns":
                    encoded = make_category_columns(data, "Transaction_Type")
                    print(encoded.head())
                case "14" | "scale_numbers":
                    columns = ["Transaction_Amount_INR", "GST_Rate", "Day_Index"]
                    print(scale_numbers(data[columns])[:5])
                case "15" | "fit_linear_model":
                    model = fit_linear_model(
                        data[["Day_Index"]], data["Transaction_Amount_INR"]
                    )
                    print(f"Amount change per day: INR {model.coef_[0]:,.2f}")
                case "16" | "gradient_descent":
                    target = data["Transaction_Amount_INR"].mean()
                    result = gradient_descent(
                        lambda value: 2 * (value - target), 0, 0.1, 100
                    )
                    print(f"Estimated average amount: INR {result:,.2f}")
                case "17" | "train_classifier":
                    features, answers = classifier_columns(data)
                    model = train_classifier(features, answers)
                    print(model)
                case "18" | "make_predictions":
                    features, answers = classifier_columns(data)
                    model = train_classifier(features, answers)
                    print(make_predictions(model, features.head()))
                case "19" | "check_accuracy":
                    features, answers = classifier_columns(data)
                    model = train_classifier(features, answers)
                    accuracy = check_accuracy(model, features, answers)
                    print(f"Training accuracy: {accuracy:.1%}")
                case "20" | "make_clusters":
                    columns = ["Transaction_Amount_INR", "Day_Index"]
                    scaled = scale_numbers(data[columns])
                    print(make_clusters(scaled, number_of_groups=3)[:20])
                case "21" | "find_frequent_items":
                    baskets = data[
                        ["Department", "Account_Head", "Payment_Mode"]
                    ].astype(str).values.tolist()
                    print(find_frequent_items(baskets, minimum_fraction=0.2))
                case "22" | "read_data_file":
                    loaded_data = read_data_file(DATA_FILE)
                    print(loaded_data.head())
                    print(f"Rows and columns: {loaded_data.shape}")
                case "23" | "make_category":
                    categories = make_category(data, "Transaction_Type")
                    print(categories["Transaction_Type"].dtype)
                case "24" | "make_matrix":
                    columns = ["Transaction_Amount_INR", "GST_Rate", "Day_Index"]
                    print(make_matrix(data, columns)[:5])
                case "25" | "column_as_list":
                    print(column_as_list(data, "Transaction_Type")[:10])
                case "26" | "summarize_column":
                    print(summarize_column(data, "Transaction_Amount_INR"))
                case "27" | "count_values":
                    print(count_values(data, "Transaction_Type"))
                case "28" | "show_preview":
                    show_preview(data)
                case "29" | "show_summary":
                    show_summary(data)
                case "30" | "check_data_quality":
                    check_data_quality(data)
                case "31" | "clean_and_save_copy":
                    clean_and_save_copy(data)
                case "32" | "save_amount_chart":
                    save_amount_chart(data)
                case "33" | "classifier_columns":
                    features, answers = classifier_columns(data)
                    print("Classifier feature columns:")
                    print(features.head())
                    print("Transaction type labels:")
                    print(answers.head())
                case "0" | "exit":
                    print("Goodbye.")
                    break
                case _:
                    print("Enter one of the listed numbers.")
        except (ImportError, KeyError, ValueError) as error:
            print(f"Could not run that function: {error}")


if __name__ == "__main__":
    main()
