# Customer Churn Prediction - Assignment 2

## a) Problem Statement

Build and deploy an end-to-end machine learning classification system to predict telecom customer churn using multiple models. The system includes model training, evaluation with standard metrics, and an interactive Streamlit interface for prediction and test-data evaluation.

## b) Dataset Description

- Dataset: Telecom Customer Churn
- Source file in this repository: telecom_customer_churn.csv
- Target column: Customer Status
- Classification type: Binary (Churn vs Stay)
- Feature count used for modeling: 18
- Instance count: More than 500 rows

## c) GitHub Repository Link

- Repository URL: https://github.com/pavanimanchala53/2025ac05763_ml_assignment/

## d) Models Used

Implemented models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)

### Comparison Table

|index|Model|Accuracy|AUC Score|Precision|Recall|F1 Score|MCC Score|Train Accuracy|
|---|---|---|---|---|---|---|---|---|
|0|Random Forest \(Ensemble\)|0\.9349|0\.9865|0\.9541|0\.9151|0\.9342|0\.8705|1\.0|
|1|Decision Tree Classifier|0\.919|0\.9194|0\.9556|0\.8805|0\.9165|0\.8407|1\.0|
|2|Logistic Regression|0\.84|0\.9172|0\.8739|0\.7987|0\.8346|0\.6829|0\.8251|
|3|K-Nearest Neighbor Classifier|0\.8268|0\.9088|0\.8782|0\.7631|0\.8166|0\.6598|0\.8753|
|4|Naive Bayes Classifier|0\.7977|0\.8805|0\.8115|0\.7809|0\.7959|0\.5959|0\.788|

### Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Served as a strong baseline with stable performance, but it was less expressive than the ensemble model. |
| Decision Tree | Captured non-linear patterns, but it is more sensitive to overfitting and less stable than the better-performing models. |
| KNN | Gave balanced results, but prediction quality depends heavily on local neighborhood structure and scaling. |
| Naive Bayes | Performed the weakest among the tested models because of its strong independence assumptions. |
| Random Forest (Ensemble) | Delivered the best overall balance of accuracy and classification quality, making it the strongest model in this dataset. |
| Overall Winner for this dataset | Random Forest (Ensemble) |

## Streamlit Application Features

- Dataset upload option (CSV test data)
- Model selection dropdown
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Single-customer prediction interface with churn/stay probabilities

## Mandatory Submission Links

1. GitHub Repository Link: https://github.com/pavanimanchala53/2025ac05763_ml_assignment/
2. Live Streamlit App Link: https://churnpredictionappml.streamlit.app/

## Repository Files

- app.py
- requirements.txt
- README.md
- churn classification.ipynb
- telecom_customer_churn.csv
- model/ (saved model files)
- scaler.pkl
- label_encoder.pkl

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

