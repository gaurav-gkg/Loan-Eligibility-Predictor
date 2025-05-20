# app/streamlit_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("D:\\Coding\\Projects\\Loan Predictor\\model\\loan_model.pkl")

# Title
st.title("🏦 Loan Eligibility Predictor")

# Input form
st.subheader("Enter Applicant Details")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Convert input to DataFrame
input_df = pd.DataFrame({
    'Gender': [0 if gender == "Male" else 1],
    'Married': [1 if married == "Yes" else 0],
    'Dependents': [3 if dependents == "3+" else int(dependents)],
    'Education': [0 if education == "Graduate" else 1],
    'Self_Employed': [1 if self_employed == "Yes" else 0],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [1 if credit_history == "Yes" else 0],
    'Property_Area': [0 if property_area == "Urban" else (1 if property_area == "Rural" else 2)],
})

# Predict
if st.button("Check Eligibility"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ You are eligible for a loan!")
    else:
        st.error("❌ You are not eligible for a loan.")
