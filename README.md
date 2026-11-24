# 🏨 Hotel Accounts Data Science Project

A simple Python project for exploring a hotel accounts dataset and practicing the main ideas from **Fundamentals of Data Science (FDS), Units I–VI**. It includes small, reusable functions, an interactive menu, analysis scripts, and a LaTeX project report.

## 📁 Project folders

| Folder | What it contains |
| --- | --- |
| `data/` | The analysis-ready hotel accounts workbook and the raw sample workbook. |
| `fds_unit_functions/` | Easy-to-read Python functions grouped by course unit. |
| `hotel_accounts_tools/` | The interactive menu for running functions on the analysis-ready workbook. |
| `analysis_scripts/` | Standalone scripts for statistics, graphs, hypothesis testing, regression, PCA, and LDA. |
| `reports/latex_source/` | LaTeX report, figures, compiled PDF, and a ZIP of the report source. |
| `project_documents/` | Project and course reference documents. |
| `outputs/` | Generated tables, charts, and cleaned data copies. |

## 🚀 Run the interactive menu

Use **Python 3.10 or newer** (the menu uses `match`/`case`). From the project root, install the packages used by the menu and unit functions:

```bash
python -m pip install pandas openpyxl matplotlib scikit-learn
```

Then start the menu:

```bash
python hotel_accounts_tools/hotel_accounts_menu.py
```

Choose a number or type a function name shown in the menu. Enter `0` to quit. The menu loads the `Analysis_Ready` sheet from `data/hotel_accounts_analysis_ready.xlsx` when it starts. Some choices display results in the terminal; chart and cleaned-copy choices save files under `outputs/hotel_accounts_tools/`.

## 🧰 Functions by unit

- **Unit I – Introduction to Data Science:** `analytics_types`, `crisp_dm_steps`, `semma_steps`
- **Unit II – Probability and Statistics:** `average`, `conditional_probability`, `correlation`, `normal_probability`
- **Unit III – Data Sources and Visualization:** `load_csv`, `describe_data`, `fill_missing`, `show_histogram`, `show_scatter_plot`
- **Unit IV – Feature Engineering and Optimization:** `make_category_columns`, `scale_numbers`, `fit_linear_model`, `gradient_descent`
- **Unit V – Machine Learning:** `train_classifier`, `make_predictions`, `check_accuracy`, `make_clusters`, `find_frequent_items`
- **Unit VI – Data Analysis Tools:** `read_data_file`, `make_category`, `make_matrix`, `column_as_list`, `summarize_column`, `count_values`

The menu also includes dataset helpers such as `show_preview`, `show_summary`, and `check_data_quality`.

## 📊 Dataset

The menu uses `hotel_accounts_analysis_ready.xlsx`, specifically its `Analysis_Ready` worksheet. A raw sample workbook is also kept in `data/`. The functions and report use fields such as transaction amount, transaction type, GST rate, department, account head, payment mode, and day index.

## 📄 Report

The report source is [`reports/latex_source/hotel_accounts_analysis_report.tex`](reports/latex_source/hotel_accounts_analysis_report.tex). The compiled report is [`reports/latex_source/hotel_accounts_analysis_report.pdf`](reports/latex_source/hotel_accounts_analysis_report.pdf), and its figures are in `reports/latex_source/figures/`.

## ✅ Notes

- The menu reads the original workbook and does not overwrite it.
- Generated files go in `outputs/`.
- Some analysis scripts may need additional Python packages depending on which script you run.
