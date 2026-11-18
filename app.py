
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from modules.dataset_analyzer import analyze_dataset
from modules.preprocessing import prepare_data
from modules.model_training import train_models
from modules.visualization import (
    plot_class_distribution,
    plot_model_comparison,
    plot_confusion_matrix
)

st.set_page_config(
    page_title="Universal AI Fraud Detection",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Universal AI Fraud Detection System")

st.write(
    "Upload a labelled fraud detection CSV dataset. "
    "The system analyzes, preprocesses, trains and evaluates models."
)

st.divider()


# Upload dataset
uploaded_file = st.file_uploader(
    "Upload Fraud Detection Dataset",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        file_size = uploaded_file.size / (1024 * 1024)

        st.info(f"Uploaded file size: {file_size:.2f} MB")

        if file_size > 500:
            st.error("File exceeds the 500 MB application limit.")
            st.stop()

        data = pd.read_csv(
            uploaded_file,
            low_memory=True,
            dtype_backend="numpy_nullable"
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(),
            use_container_width=True
        )

        clean_data, info = analyze_dataset(data)

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", info["rows"])
        col2.metric("Total Columns", info["columns"])
        col3.metric("Fraud Target", info["target"])

        st.write("Detected Features:", info["features"])

        st.subheader("Class Distribution")

        fig = plot_class_distribution(
            clean_data[info["target"]]
        )

        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)


        # Train models
        if st.button("🚀 Train Models and Detect Fraud"):

            with st.spinner(
                "Preprocessing dataset and training models..."
            ):

                X, y, preprocessor = prepare_data(
                    clean_data,
                    info["target"]
                )

                output = train_models(
                    X,
                    y,
                    preprocessor
                )

            st.success("Model training completed!")

            st.subheader("Best Model")

            st.success(
                f"Selected Model: {output['best_model_name']}"
            )


            # Model performance
            st.subheader("Model Performance")

            results_df = pd.DataFrame(
                output["results"]
            ).T

            st.dataframe(
                results_df.style.format("{:.4f}"),
                use_container_width=True
            )


            # Model comparison
            st.subheader("Model Comparison")

            fig = plot_model_comparison(
                output["results"]
            )

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)


            # Confusion matrix
            st.subheader("Confusion Matrix")

            fig = plot_confusion_matrix(
                output["confusion_matrix"]
            )

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)


            # Prediction results
            st.subheader("Prediction Results")

            predictions = output["predictions"]

            probabilities = output["best_model"].predict_proba(
                output["X_test"]
            )[:, 1]

            results = output["X_test"].copy()

            results["Actual"] = output["y_test"].values

            results["Prediction"] = predictions

            results["Fraud Probability (%)"] = (
                probabilities * 100
            ).round(2)

            results["Prediction Label"] = results[
                "Prediction"
            ].map({
                0: "Normal",
                1: "Fraud"
            })

            st.dataframe(
                results.head(100),
                use_container_width=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Prediction Results",
                data=csv_data,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )


            # SHAP Explainable AI
            st.divider()

            st.header("🔍 Explainable AI (SHAP)")

            st.write(
                "SHAP explains which features influence "
                "the model's fraud predictions."
            )

            shap_values = output["shap_values"]
            shap_data = output["shap_data"]

            feature_names = output["shap_feature_names"]

            # Global feature importance
            st.subheader("Feature Importance")

            importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(
                "Importance",
                ascending=False
            ).head(10)

            fig, ax = plt.subplots(figsize=(6, 3.5))

            ax.barh(
                importance_df["Feature"][::-1],
                importance_df["Importance"][::-1]
            )

            ax.set_title("Top 10 Important Features")
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.tick_params(axis="y", labelsize=8)

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)


            # Individual transaction explanation
            st.subheader("Individual Transaction Explanation")

            if len(shap_data) > 0:

                selected_index = st.selectbox(
                    "Select transaction to explain",
                    options=list(range(len(shap_data))),
                    format_func=lambda i: (
                        f"Transaction {i + 1}"
                    )
                )

                selected_values = shap_values[selected_index]

                explanation_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": selected_values
                })

                explanation_df["Impact"] = np.where(
                    explanation_df["SHAP Value"] > 0,
                    "Pushes towards Fraud",
                    "Pushes towards Normal"
                )

                explanation_df["Absolute Impact"] = (
                    explanation_df["SHAP Value"].abs()
                )

                explanation_df = explanation_df.sort_values(
                    "Absolute Impact",
                    ascending=False
                ).head(10)

                st.write(
                    "Features with the strongest influence "
                    "on this transaction:"
                )

                st.dataframe(
                    explanation_df[
                        ["Feature", "SHAP Value", "Impact"]
                    ].round(4),
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(6, 3.5))

                ax.barh(
                    explanation_df["Feature"][::-1],
                    explanation_df["SHAP Value"][::-1]
                )

                ax.axvline(0, linewidth=1)

                ax.set_title(
                    "Individual Transaction SHAP Explanation"
                )

                ax.set_xlabel("SHAP Value")
                ax.tick_params(axis="y", labelsize=8)

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=False
                )

                plt.close(fig)

            st.success(
                "Training, prediction and SHAP explanation completed."
            )

    except Exception as e:

        st.error(f"Dataset processing error: {e}")

st.divider()

st.caption(
    "Universal AI Fraud Detection System | "
    "Machine Learning Based Classification"
)
