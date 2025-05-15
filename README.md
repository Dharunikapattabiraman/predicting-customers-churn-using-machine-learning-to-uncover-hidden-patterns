# predicting-customers-churn-using-machine-learning-to-uncover-hidden-patterns
# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and prepare the data
df = pd.read_csv("botswana_bank_customer_churn.csv")
df = df.drop(columns=[
    'RowNumber', 'CustomerId', 'Surname', 'First Name', 'Date of Birth',
    'Address', 'Contact Information', 'Churn Reason', 'Churn Date'
])
df = df.dropna()

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Prepare features and target
X = df.drop('Churn Flag', axis=1)
y = df['Churn Flag']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Streamlit App
st.title("Botswana Bank Customer Churn Prediction")

# User inputs
user_input = {}
for col in X.columns:
    if col in categorical_cols:
        options = list(encoders[col].classes_)
        selection = st.selectbox(f"{col}", options)
        user_input[col] = encoders[col].transform([selection])[0]
    else:
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = "Customer is likely to churn." if prediction == 1 else "Customer is likely to stay."
    st.success(result)
