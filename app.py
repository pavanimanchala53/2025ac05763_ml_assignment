import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def get_root_dir() -> Path:
    # __file__ can be missing in notebook/IPython contexts.
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


ROOT_DIR = get_root_dir()

FEATURES = [
    "Age",
    "Married",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Internet Service",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charge",
    "Total Charges",
    "Total Long Distance Charges",
    "Total Revenue",
]

CATEGORICAL_COLS = [
    "Married",
    "Internet Service",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]


def running_in_streamlit() -> bool:
    return get_script_run_ctx() is not None


def find_artifact(filename: str) -> Path | None:
    candidates = [
        ROOT_DIR / "model" / filename,
        Path.cwd() / "model" / filename,
        ROOT_DIR / filename,
        Path.cwd() / filename,
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def load_available_models() -> tuple[dict[str, object], dict[str, str]]:
    model_file_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree_classifier.pkl",
        "KNN": "k_nearest_neighbor_classifier.pkl",
        "Naive Bayes": "naive_bayes_classifier.pkl",
        "Random Forest": "random_forest_ensemble.pkl",
        "Default Model": "model.pkl",
    }

    loaded_models: dict[str, object] = {}
    load_errors: dict[str, str] = {}
    for model_name, file_name in model_file_map.items():
        model_path = find_artifact(file_name)
        if model_path is not None:
            try:
                with open(model_path, "rb") as f:
                    loaded_models[model_name] = pickle.load(f)
            except Exception as exc:
                load_errors[model_name] = f"{type(exc).__name__}: {exc}"

    return loaded_models, load_errors


def run_app() -> None:
    label_path = find_artifact("label_encoder.pkl")
    scaler_path = find_artifact("scaler.pkl")
    available_models, model_load_errors = load_available_models()

    missing_files = []
    if label_path is None:
        missing_files.append("label_encoder.pkl")
    if scaler_path is None:
        missing_files.append("scaler.pkl")

    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info(f"Checked locations: {ROOT_DIR} and {Path.cwd()}")
        return

    if not available_models:
        st.error("No usable model files found.")
        st.info("Expected at least one of: model.pkl, logistic_regression.pkl, decision_tree_classifier.pkl, k_nearest_neighbor_classifier.pkl, naive_bayes_classifier.pkl, random_forest_ensemble.pkl")
        if model_load_errors:
            st.error("Model load errors:")
            for model_name, error_text in model_load_errors.items():
                st.write(f"- {model_name}: {error_text}")
        return

    if model_load_errors:
        st.warning("Some model files could not be loaded in this environment. Using only available models.")
        with st.expander("Show model load details"):
            for model_name, error_text in model_load_errors.items():
                st.write(f"- {model_name}: {error_text}")

    try:
        with open(label_path, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as exc:
        st.error(f"Failed to load label_encoder.pkl: {type(exc).__name__}: {exc}")
        return

    try:
        with open(scaler_path, "rb") as f:
            sc = pickle.load(f)
    except Exception as exc:
        st.error(f"Failed to load scaler.pkl: {type(exc).__name__}: {exc}")
        return

    def categorical_options(col: str, fallback: list[str]) -> list[str]:
        encoder = label_encoder.get(col)
        if encoder is not None and hasattr(encoder, "classes_"):
            return [str(v) for v in encoder.classes_.tolist()]
        return fallback

    married_choices = categorical_options("Married", ["Yes", "No"])
    internet_choices = categorical_options("Internet Service", ["Yes", "No"])
    online_backup_choices = categorical_options("Online Backup", ["Yes", "No"])
    device_plan_choices = categorical_options("Device Protection Plan", ["Yes", "No"])
    tech_support_choices = categorical_options("Premium Tech Support", ["Yes", "No"])
    stream_tv_choices = categorical_options("Streaming TV", ["Yes", "No"])
    stream_movies_choices = categorical_options("Streaming Movies", ["Yes", "No"])
    contract_choices = categorical_options("Contract", ["Month-to-Month", "One Year", "Two Year"])
    paperless_choices = categorical_options("Paperless Billing", ["Yes", "No"])
    payment_choices = categorical_options("Payment Method", ["Credit Card", "Cash/Online"])

    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
            .css-18e3th9 {padding-top: 1rem;}
            .center-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
            }
        </style>
        <h1 class="center-title">Customer Churn Prediction</h1>
        """,
        unsafe_allow_html=True,
    )

    def preprocess_input_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        missing_cols = [col for col in FEATURES if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        frame = df[FEATURES].copy()

        for col in CATEGORICAL_COLS:
            if col not in label_encoder:
                raise KeyError(f"Label encoder for '{col}' not found")
            try:
                frame[col] = label_encoder[col].transform(frame[col])
            except ValueError as exc:
                raise ValueError(f"Invalid value in column '{col}': {exc}") from exc

        if hasattr(sc, "feature_names_in_"):
            frame = frame.reindex(columns=list(sc.feature_names_in_))

        frame_scaled = sc.transform(frame)
        return frame, frame_scaled

    def preprocess_single_input(data: list) -> np.ndarray:
        df = pd.DataFrame([data], columns=FEATURES)
        _, x_data = preprocess_input_frame(df)
        return x_data

    def normalize_target_labels(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            values = set(pd.Series(series).dropna().astype(int).unique().tolist())
            if values.issubset({0, 1}):
                return series.astype(int)

        mapped: list[int] = []
        for value in series.astype(str):
            text = value.strip().lower()
            if "churn" in text:
                mapped.append(0)
            elif "stay" in text:
                mapped.append(1)
            elif text in {"0", "1"}:
                mapped.append(int(text))
            else:
                raise ValueError(
                    "Customer Status values must be churn/stay or 0/1."
                )

        return pd.Series(mapped, index=series.index)

    def predict_customer_status(data: list, selected_model_name: str):
        x_data = preprocess_single_input(data)
        selected_model = available_models[selected_model_name]
        probs = selected_model.predict_proba(x_data)[0]

        class_labels = list(selected_model.classes_) if hasattr(selected_model, "classes_") else [0, 1]
        prob_by_class = dict(zip(class_labels, probs))
        churn_prob = float(prob_by_class.get(0, probs[0]))
        stay_prob = float(prob_by_class.get(1, probs[-1]))
        status_val = "Stay" if stay_prob >= churn_prob else "Churn"

        return status_val, stay_prob, churn_prob

    def evaluate_uploaded_test_data(uploaded_df: pd.DataFrame, selected_model_name: str) -> None:
        target_col = "Customer Status"
        if target_col not in uploaded_df.columns:
            st.error("Uploaded CSV must include 'Customer Status' column.")
            return

        eval_df = uploaded_df.copy()
        # Drop rows that cannot be mapped to churn/stay labels.
        allowed_mask = eval_df[target_col].astype(str).str.lower().str.contains(
            "churn|stay|0|1", regex=True
        )
        filtered_df = eval_df[allowed_mask].copy()

        if filtered_df.empty:
            st.error("No valid labeled rows found after filtering target labels.")
            return

        try:
            y_true = normalize_target_labels(filtered_df[target_col])
            _, x_eval = preprocess_input_frame(filtered_df)
            model = available_models[selected_model_name]
            y_pred = model.predict(x_eval)
            y_proba = model.predict_proba(x_eval)
        except Exception as exc:
            st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
            return

        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        prob_map = {label: y_proba[:, idx] for idx, label in enumerate(classes)}
        stay_scores = prob_map.get(1, y_proba[:, -1])

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        auc_score = np.nan
        if y_true.nunique() == 2:
            auc_score = roc_auc_score(y_true, stay_scores)

        st.subheader("Test Data Evaluation")
        m1, m2, m3 = st.columns(3)
        m4, m5, m6 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy:.4f}")
        m2.metric("AUC Score", "N/A" if np.isnan(auc_score) else f"{auc_score:.4f}")
        m3.metric("Precision", f"{precision:.4f}")
        m4.metric("Recall", f"{recall:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Churn (0)", "Actual Stay (1)"],
            columns=["Predicted Churn (0)", "Predicted Stay (1)"],
        )
        st.subheader("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)

        if len(filtered_df) != len(eval_df):
            st.info(
                f"Rows used for evaluation: {len(filtered_df)} / {len(eval_df)} "
                "(rows with unsupported target labels were ignored)."
            )

    col1, spacer, col2 = st.columns([1, 0.1, 2])

    with col1:
        with st.form("customer_form"):
            selected_model_name = st.selectbox("Select Model", list(available_models.keys()))
            age = st.number_input("Age", min_value=0)
            married = st.selectbox("Married", married_choices)
            dependents = st.number_input("Number of Dependents", min_value=0)
            referrals = st.number_input("Number of Referrals", min_value=0)
            tenure = st.number_input("Tenure in Months", min_value=0)
            internet = st.selectbox("Internet Service", internet_choices)
            online_backup = st.selectbox("Online Backup", online_backup_choices)
            device_plan = st.selectbox("Device Protection Plan", device_plan_choices)
            tech_support = st.selectbox("Premium Tech Support", tech_support_choices)
            stream_tv = st.selectbox("Streaming TV", stream_tv_choices)
            stream_movies = st.selectbox("Streaming Movies", stream_movies_choices)
            contract = st.selectbox("Contract", contract_choices)
            paperless = st.selectbox("Paperless Billing", paperless_choices)
            payment = st.selectbox("Payment Method", payment_choices)
            monthly = st.number_input("Monthly Charge", min_value=0.0)
            total = st.number_input("Total Charges", min_value=0.0)
            long_dist = st.number_input("Total Long Distance Charges", min_value=0.0)
            revenue = st.number_input("Total Revenue", min_value=0.0)
            submitted = st.form_submit_button("Predict")

        uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])

    with col2:
        status_box = st.empty()
        prob_box = st.empty()

    if submitted:
        input_data = [
            age,
            married,
            dependents,
            referrals,
            tenure,
            internet,
            online_backup,
            device_plan,
            tech_support,
            stream_tv,
            stream_movies,
            contract,
            paperless,
            payment,
            monthly,
            total,
            long_dist,
            revenue,
        ]

        try:
            status_val, stay_prob, churn_prob = predict_customer_status(input_data, selected_model_name)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            return

        status_box.markdown(
            f"""
            <div style='display:flex; justify-content:center; align-items:center; height:60px;'>
                <span style='font-size:32px; font-weight:bold; color:{"red" if status_val=="Churn" else "green"}'>
                {status_val}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prob_box.markdown(
            f"""
            <div>Model: {selected_model_name}</div>
            <div>Probability of Stay: {stay_prob:.2f}</div>
            <div style='background:#f0f0f0; border-radius:3px; height:5px; margin:2px 0 10px 0;'>
                <div style='width:{stay_prob*100:.1f}%; background:green; height:100%;'></div>
            </div>

            <div>Probability of Churn: {churn_prob:.2f}</div>
            <div style='background:#f0f0f0; border-radius:3px; height:5px; margin:2px 0 10px 0;'>
                <div style='width:{churn_prob*100:.1f}%; background:red; height:100%;'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {type(exc).__name__}: {exc}")
            return

        evaluate_uploaded_test_data(uploaded_df, selected_model_name)



if running_in_streamlit():
    run_app()
elif __name__ == "__main__":
    app_path = ROOT_DIR / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)
