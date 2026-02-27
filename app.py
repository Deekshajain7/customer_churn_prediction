import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn or stay.")

# Sidebar model info
st.sidebar.header("About")
st.sidebar.write("This app predicts customer churn using Machine Learning.")

st.sidebar.write("Model Used: Random Forest")

# -----------------------
# User Input Section
# -----------------------

st.subheader("Enter Customer Details")

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

# -----------------------
# Manual Encoding
# -----------------------

def user_input_features():
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    df = pd.DataFrame([data])

    return df


input_df = user_input_features()

# Load original dataset to match dummy columns
original_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
original_df.drop("customerID", axis=1, inplace=True)
original_df["TotalCharges"] = pd.to_numeric(original_df["TotalCharges"], errors="coerce")
original_df["TotalCharges"].fillna(original_df["TotalCharges"].median(), inplace=True)
original_df["Churn"] = original_df["Churn"].map({"Yes": 1, "No": 0})
original_df = pd.get_dummies(original_df, drop_first=True)

model_columns = original_df.drop("Churn", axis=1).columns

# Create empty dataframe with all columns
input_encoded = pd.DataFrame(columns=model_columns)
input_encoded.loc[0] = 0

# Fill known numeric values
input_encoded["tenure"] = tenure
input_encoded["MonthlyCharges"] = monthly_charges
input_encoded["TotalCharges"] = total_charges

# Scale
input_scaled = scaler.transform(input_encoded)

# -----------------------
# Prediction
# -----------------------

if st.button("Predict"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN")
    else:
        st.success(f"✅ Customer is likely to STAY")

    st.write(f"Churn Probability: {round(probability * 100, 2)}%")