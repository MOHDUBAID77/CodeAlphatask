import os, joblib
import pandas as pd 
import seaborn as sns 
import streamlit as st 

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Disease Prediction from Medical Data")
st.caption("Educational machine-learning demonstration — not a medical diagnosis.")

models = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "SVM": "models/svm.joblib",
    "Random Forest": "models/random_forest.joblib",
    "XGBoost": "models/xgboost.joblib"
}
available = {k:v for k,v in models.items() if os.path.exists(v)}
if not available:
    st.warning("No trained models found. Run `python train.py` first.")
    st.stop()

name = st.sidebar.selectbox("Choose model", list(available))
bundle = joblib.load(available[name])
pipe = bundle["pipeline"]
features = bundle["feature_columns"]
classes = bundle["target_classes"]

uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
    missing = [c for c in features if c not in data.columns]
    if missing:
        st.error(f"Missing feature columns: {missing}")
        st.stop()
    pred = pipe.predict(data[features])
    data["prediction"] = [classes[int(p)] for p in pred]
    if hasattr(pipe, "predict_proba"):
        data["prediction_confidence"] = pipe.predict_proba(data[features]).max(axis=1)
    st.dataframe(data, use_container_width=True)
    st.download_button("Download predictions", data.to_csv(index=False),
                       "disease_predictions.csv", "text/csv")
else:
    st.info("Upload a CSV with the same feature columns used during training.")

st.divider()
st.write("Workflow: CSV → preprocessing → four classifiers → evaluation → prediction")
