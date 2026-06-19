import streamlit as st
import joblib
import numpy as np
import pandas as pd


# load model
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")

st.sidebar.info("""
### Project Overview

This project uses a Random Forest model trained on SMOTE-balanced data to detect fraudulent credit card transactions.

Technologies Used:
- Python
- Scikit-Learn
- Random Forest
- SMOTE
- Streamlit
- Power BI
""")

st.write(
    "Predict whether a transaction is fraudulent or genuine."
)

# ===== KPI METRICS =====
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.95%")
col2.metric("Precision", "91%")
col3.metric("Recall", "75%")
col4.metric("ROC-AUC", "95.2%")

st.subheader("Enter Transaction Details")


with st.expander("Manual Prediction (Advanced)"):

    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    v_features = []

    for i in range(1,29):
        value = st.number_input(f"V{i}", value=0.0)
        v_features.append(value)

    if st.button("Predict"):

        scaled_time = scaler.transform([[time]])[0][0]
        scaled_amount = scaler.transform([[amount]])[0][0]

        input_data = np.array(
            [[scaled_time] + v_features + [scaled_amount]]
        )

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction")
            st.snow()
        else:
            st.success("✅ Genuine Transaction")
            st.balloons()

        st.write(f"Fraud Probability: {probability:.2%}")


st.subheader("Confusion Matrix")
st.image("confusion_matrix.png")

st.subheader("ROC Curve")
st.image("ROC_curve.png")

st.subheader("Power BI Dashboard")
st.image("dashboard.png")