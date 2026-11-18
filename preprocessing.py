import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def prepare_data(df, target_column):

    data = df.copy()

    # Remove rows with missing target
    data = data.dropna(subset=[target_column])

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column].copy()

    # Binary classification only
    if y.nunique() != 2:
        raise ValueError(
            "Fraud detection requires exactly two target classes."
        )

    # Convert target into 0 and 1
    values = list(y.unique())

    fraud_names = [
        "1", "fraud", "fraudulent",
        "yes", "true", "positive"
    ]

    positive = next(
        (v for v in values
         if str(v).strip().lower() in fraud_names),
        None
    )

    if positive is None:
        try:
            numeric_values = pd.to_numeric(y)
            if set(numeric_values.unique()) == {0, 1}:
                positive = next(
                    v for v in values if float(v) == 1
                )
            else:
                positive = y.value_counts().idxmin()
        except (ValueError, TypeError):
            positive = y.value_counts().idxmin()

    y = (y == positive).astype(int)

    # Remove identifier-like columns
    id_names = [
        "id", "transactionid", "transaction_id",
        "nameorig", "namedest"
    ]

    remove_columns = [
        col for col in X.columns
        if str(col).strip().lower().replace(" ", "")
        in id_names
    ]

    X = X.drop(columns=remove_columns, errors="ignore")

    # Remove constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if X.empty:
        raise ValueError("No useful features found.")

    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline,
             numerical_columns),

            ("categorical", categorical_pipeline,
             categorical_columns)
        ]
    )

    return X, y, preprocessor