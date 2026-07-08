from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


APP_TITLE = "Loan Approval Predictor"
MODEL_PATH = Path(__file__).with_name("loan_approval_model.pkl")


st.set_page_config(page_title=APP_TITLE, page_icon="🏦", layout="centered")
st.title(APP_TITLE)
st.write("Enter applicant details to predict whether the loan is approved or denied.")


@st.cache_resource
def load_bundle():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        f"Missing model file: {MODEL_PATH.name}. Run the notebook cell that saves the pickle first.")
    st.stop()

model = bundle["model"] if isinstance(bundle, dict) else bundle[0]
scaler = bundle["scaler"] if isinstance(bundle, dict) else bundle[1]
feature_columns = bundle.get("feature_columns") if isinstance(
    bundle, dict) else bundle[2]

with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18,
                              max_value=100, value=30, step=1)
        annual_income = st.number_input(
            "Annual Income", min_value=0, value=50000, step=1000)
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650, step=1)

    with col2:
        loan_amount = st.number_input(
            "Loan Amount", min_value=0, value=15000, step=1000)
        employment_years = st.number_input(
            "Employment Years", min_value=0, max_value=50, value=5, step=1)

    submitted = st.form_submit_button("Predict")


if submitted:
    applicant = pd.DataFrame(
        [{
            "Age": age,
            "Annual_Income": annual_income,
            "Credit_Score": credit_score,
            "Loan_Amount": loan_amount,
            "Employment_Years": employment_years,
        }]
    )

    if feature_columns:
        applicant = applicant[feature_columns]

    applicant_scaled = scaler.transform(applicant)
    prediction = int(model.predict(applicant_scaled)[0])
    probability = float(model.predict_proba(applicant_scaled)[0][1])

    st.subheader("Result")
    if prediction == 1:
        st.success("Approved")
    else:
        st.error("Denied")

    st.metric("Approval Probability", f"{probability:.2%}")
    st.caption(
        "The probability shown is the model's estimated chance of approval.")
