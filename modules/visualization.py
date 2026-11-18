import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(y).value_counts().plot(
        kind="bar", ax=ax
    )
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Transactions")
    plt.tight_layout()
    return fig


def plot_model_comparison(results):
    df = pd.DataFrame(results).T

    fig, ax = plt.subplots(figsize=(9, 5))
    df[["Precision", "Recall", "F1 Score"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig
