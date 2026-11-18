AI Credit Card Fraud Detection System

Project Overview

The AI Credit Card Fraud Detection System is a machine learning-based application designed to identify potentially fraudulent credit card transactions. It analyzes uploaded transaction datasets and uses machine learning algorithms to classify transactions as legitimate or fraudulent.

Features

- Upload CSV transaction datasets.
- Data preprocessing and analysis.
- Fraud detection using machine learning.
- Model training and comparison.
- Fraud distribution visualization.
- Confusion matrix visualization.
- Performance evaluation using multiple metrics.

Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

Machine Learning Models

- Logistic Regression
- Random Forest
- XGBoost

Project Structure

- "app.py" – Main Streamlit application.
- "train_model.py" – Model training script.
- "modules/" – Data analysis, preprocessing, training and visualization modules.
- "results/" – Trained model and model comparison results.
- ".streamlit/" – Streamlit configuration.
- "requirements.txt" – Project dependencies.

Installation

1. Clone the repository:

git clone YOUR_REPOSITORY_URL

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

streamlit run app.py

Purpose

The purpose of this project is to demonstrate how machine learning techniques can be used to analyze credit card transactions and support fraud detection.

Author

Aisha Khalid

License

This project is developed for educational purposes.