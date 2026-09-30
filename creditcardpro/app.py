import joblib
import pandas as pd 
import seaborn as sns 
import streamlit as st 

st.set_page_config(
    page_title="Credit Scoring Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Scoring Predictor")
st.write("Enter financial information to estimate creditworthiness.")

bundle = joblib.load("models/credit_scoring_model.pkl")
model = bundle["model"]
features = bundle["features"]

income = st.number_input("Annual Income", min_value=1000.0, value=50000.0)
debt = st.number_input("Total Debt", min_value=0.0, value=15000.0)
credit_age_years = st.number_input("Credit Age (years)", min_value=0.0, value=5.0)
late_payments = st.number_input("Late Payments", min_value=0, value=1)
total_payments = st.number_input("Total Payments", min_value=1, value=60)
credit_utilization = st.slider("Credit Utilization", 0.01, 1.0, 0.35)
loan_count = st.number_input("Number of Loans", min_value=0, value=2)
savings = st.number_input("Savings", min_value=0.0, value=30000.0)

if st.button("Predict Creditworthiness"):
    debt_to_income = debt / max(income, 1)
    payment_success_rate = 1 - late_payments / max(total_payments, 1)
    available_savings_ratio = savings / max(income, 1)

    data = pd.DataFrame([{
        "income": income,
        "debt": debt,
        "credit_age_years": credit_age_years,
        "late_payments": late_payments,
        "total_payments": total_payments,
        "credit_utilization": credit_utilization,
        "loan_count": loan_count,
        "savings": savings,
        "debt_to_income": debt_to_income,
        "payment_success_rate": payment_success_rate,
        "available_savings_ratio": available_savings_ratio
    }])[features]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success("Prediction: Creditworthy")
    else:
        st.warning("Prediction: Higher Credit Risk")

    st.metric("Estimated Creditworthiness Probability", f"{probability:.2%}")

st.caption(
    "Educational project only. This prediction should not be used as a real lending decision."
)
