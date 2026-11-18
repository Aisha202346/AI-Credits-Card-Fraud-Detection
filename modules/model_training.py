
import os
import joblib
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def train_models(X, y, preprocessor, results_dir="results"):

    os.makedirs(results_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC AUC": roc_auc_score(y_test, probabilities)
        }

        trained_models[name] = pipeline

    best_name = max(
        results,
        key=lambda name: results[name]["F1 Score"]
    )

    best_model = trained_models[best_name]

    # Save best model
    joblib.dump(
        best_model,
        os.path.join(results_dir, "best_model.pkl")
    )

    # Save model comparison
    pd.DataFrame(results).T.to_csv(
        os.path.join(results_dir, "model_comparison.csv")
    )

    # Evaluate best model
    best_predictions = best_model.predict(X_test)

    cm = confusion_matrix(y_test, best_predictions)

    # SHAP preparation
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    trained_classifier = best_model.named_steps["classifier"]

    # Transform test data using fitted preprocessing
    X_test_transformed = fitted_preprocessor.transform(X_test)

    # Get transformed feature names
    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [
            f"Feature_{i}"
            for i in range(X_test_transformed.shape[1])
        ]

    # Convert transformed data into DataFrame
    X_shap = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index
    )

    # Use a small sample for faster explanations
    X_shap_sample = X_shap.head(100)

    # Generate SHAP explainer
    if best_name == "Random Forest":

        explainer = shap.TreeExplainer(trained_classifier)

        shap_values = explainer.shap_values(X_shap_sample)

        # Select fraud class (class 1)
        if isinstance(shap_values, list):
            fraud_shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            fraud_shap_values = shap_values[:, :, 1]
        else:
            fraud_shap_values = shap_values

    else:

        explainer = shap.LinearExplainer(
            trained_classifier,
            X_shap
        )

        shap_values = explainer.shap_values(X_shap_sample)

        if isinstance(shap_values, list):
            fraud_shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            fraud_shap_values = shap_values[:, :, 1]
        else:
            fraud_shap_values = shap_values

    # Return everything needed by the dashboard
    return {
        "best_model": best_model,
        "best_model_name": best_name,
        "results": results,
        "confusion_matrix": cm,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": best_predictions,
        "shap_values": fraud_shap_values,
        "shap_data": X_shap_sample,
        "shap_feature_names": list(feature_names)
    }
