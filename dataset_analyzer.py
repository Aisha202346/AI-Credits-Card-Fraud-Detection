import pandas as pd

# Supported fraud target column names
TARGET_NAMES = [
    "class",
    "isfraud",
    "fraud",
    "is_fraud",
    "fraudulent",
    "label",
    "target"
]


def analyze_dataset(df):

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Normalize column names for detection
    column_map = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    # Detect target column
    target_column = None

    for name in TARGET_NAMES:
        if name in column_map:
            target_column = column_map[name]
            break

    if target_column is None:
        raise ValueError(
            "Fraud target column not found. "
            "Supported names: Class, isFraud, fraud, label, target."
        )

    # Check target values
    target_values = df[target_column].dropna().unique()

    if len(target_values) < 2:
        raise ValueError(
            "Target column must contain at least two classes."
        )

    if len(target_values) > 20:
        raise ValueError(
            "Too many target classes for binary fraud detection."
        )

    # Remove rows with missing target
    clean_df = df.dropna(subset=[target_column]).copy()

    # Identify features
    feature_columns = [
        col for col in clean_df.columns
        if col != target_column
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    # Basic dataset information
    information = {
        "rows": len(clean_df),
        "columns": len(clean_df.columns),
        "target": target_column,
        "features": feature_columns,
        "classes": clean_df[target_column].nunique(),
        "missing_values": int(clean_df.isnull().sum().sum()),
        "duplicates": int(clean_df.duplicated().sum())
    }

    return clean_df, information


def get_dataset_report(df):

    clean_df, info = analyze_dataset(df)

    report = {
        "Dataset Rows": info["rows"],
        "Dataset Columns": info["columns"],
        "Target Column": info["target"],
        "Number of Features": len(info["features"]),
        "Target Classes": info["classes"],
        "Missing Values": info["missing_values"],
        "Duplicate Rows": info["duplicates"]
    }

    return report