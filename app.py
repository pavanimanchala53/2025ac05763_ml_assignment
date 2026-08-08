import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def get_root_dir() -> Path:
    # __file__ can be missing in notebook/IPython contexts.
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


ROOT_DIR = get_root_dir()


def running_in_streamlit() -> bool:
    return get_script_run_ctx() is not None


def find_artifact(filename: str) -> Path | None:
    candidates = [
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

    features = [
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

    def preprocess_input(data: list) -> tuple[pd.DataFrame, np.ndarray]:
        df = pd.DataFrame([data], columns=features)
        cols = [
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
        for col in cols:
            df[col] = label_encoder[col].transform(df[col])
        if hasattr(sc, "feature_names_in_"):
            df = df.reindex(columns=list(sc.feature_names_in_))
        df_scaled = sc.transform(df)
        return df, df_scaled

    def predict_customer_status(data: list, selected_model_name: str):
        _, x_data = preprocess_input(data)
        selected_model = available_models[selected_model_name]
        probs = selected_model.predict_proba(x_data)[0]

        class_labels = list(selected_model.classes_) if hasattr(selected_model, "classes_") else [0, 1]
        prob_by_class = dict(zip(class_labels, probs))
        churn_prob = float(prob_by_class.get(0, probs[0]))
        stay_prob = float(prob_by_class.get(1, probs[-1]))
        status_val = "Stay" if stay_prob >= churn_prob else "Churn"

        return status_val, stay_prob, churn_prob

    col1, spacer, col2 = st.columns([1, 0.1, 2])

    with col1:
        with st.form("customer_form"):
            selected_model_name = st.selectbox("Select Model", list(available_models.keys()))
            age = st.number_input("Age", min_value=0)
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.number_input("Number of Dependents", min_value=0)
            referrals = st.number_input("Number of Referrals", min_value=0)
            tenure = st.number_input("Tenure in Months", min_value=0)
            internet = st.selectbox("Internet Service", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            device_plan = st.selectbox("Device Protection Plan", ["Yes", "No"])
            tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
            stream_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            stream_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Credit Card", "Cash/Online"])
            monthly = st.number_input("Monthly Charge", min_value=0.0)
            total = st.number_input("Total Charges", min_value=0.0)
            long_dist = st.number_input("Total Long Distance Charges", min_value=0.0)
            revenue = st.number_input("Total Revenue", min_value=0.0)
            submitted = st.form_submit_button("Predict")

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



if running_in_streamlit():
    run_app()
elif __name__ == "__main__":
    app_path = ROOT_DIR / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)
