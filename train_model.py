# ==========================================
# AI CREDIT CARD FRAUD DETECTION PROJECT
# ==========================================

# STEP 1: IMPORT LIBRARIES

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# STEP 2: LOAD DATASET

print("\nLOADING DATASET...")

df = pd.read_csv("Dataset/creditcard.csv")

print("Dataset Shape:", df.shape)


# STEP 3: EXPLORE DATASET

print("\nFIRST FIVE ROWS:")
print(df.head())

print("\nDATASET INFORMATION:")
df.info()

print("\nMISSING VALUES:")
print(df.isnull().sum().sum())

print("\nFRAUD DISTRIBUTION:")
print(df["Class"].value_counts())


# STEP 4: CHECK AND REMOVE DUPLICATES

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nCLEANED DATASET SHAPE:")
print(df.shape)


# STEP 5: VISUALIZATION

plt.figure(figsize=(8, 5))

sns.countplot(x="Class", data=df)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.savefig("fraud_distribution.png")
plt.show()


plt.figure(figsize=(10, 5))

sns.histplot(data=df, x="Amount", bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("amount_distribution.png")
plt.show()


# STEP 6: SEPARATE FEATURES AND TARGET

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFEATURES SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)


# STEP 7: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING SHAPE:", X_train.shape)
print("TESTING SHAPE:", X_test.shape)


# STEP 8: FEATURE SCALING

preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough"
)


# STEP 9: DEFINE MACHINE LEARNING MODELS

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# STEP 10: CREATE MODELS FOLDER

os.makedirs("models", exist_ok=True)


# STEP 11: TRAIN AND EVALUATE MODELS

results = {}
trained_models = {}

for name, classifier in models.items():

    print("\n================================")
    print("TRAINING:", name)
    print("================================")

    model = Pipeline(steps=[
        ("scaler", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc
    }

    trained_models[name] = model

    print("\nACCURACY:", accuracy)
    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F1 SCORE:", f1)
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))


# STEP 12: MODEL COMPARISON

results_df = pd.DataFrame(results).T

print("\nMODEL PERFORMANCE COMPARISON:")
print(results_df)

results_df.to_csv("models/model_comparison.csv")


# STEP 13: SELECT BEST MODEL

best_model_name = results_df["F1 Score"].idxmax()

best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)


# STEP 14: SAVE BEST MODEL

joblib.dump(best_model, "models/best_model.pkl")

print("\nBEST MODEL SAVED!")


# STEP 15: SAVE FEATURE NAMES

with open("models/features.json", "w") as file:
    json.dump(list(X.columns), file)


# STEP 16: SAVE MODEL METRICS

with open("models/metrics.json", "w") as file:
    json.dump(results, file, indent=4)


# STEP 17: CONFUSION MATRIX GRAPH FOR BEST MODEL

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Best Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()


# FINAL MESSAGE

print("\n================================")
print("PROJECT TRAINING COMPLETED!")
print("================================")

print("Best Model:", best_model_name)
print("Saved: models/best_model.pkl")
print("All Models Evaluated Successfully!")
