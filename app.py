import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("medical_cost_model.pkl")

st.title("🏥 Medical Cost Prediction App")

st.write("Enter patient details")

# ------------------------------
# Inputs
# ------------------------------
age = st.number_input("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ------------------------------
# Encoding (must match training)
# ------------------------------
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}
region = region_map[region]

# ------------------------------
# Feature Engineering (FROM TRAINING)
# ------------------------------
bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3

age_group = 0 if age < 25 else 1 if age < 40 else 2 if age < 60 else 3

smoker_bmi = smoker * bmi
smoker_age = smoker * age

# ------------------------------
# FINAL INPUT (EXACT ORDER)
# ------------------------------
input_data = pd.DataFrame([[
    age,
    sex,
    bmi,
    children,
    smoker,
    region,
    bmi_category,
    age_group,
    smoker_bmi,
    smoker_age
]], columns=[
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "bmi_category",
    "age_group",
    "smoker_bmi",
    "smoker_age"
])

# ------------------------------
# Debug
# ------------------------------
st.write("Model loaded:", os.path.exists("medical_cost_model.pkl"))
st.write("Input shape:", input_data.shape)

# ------------------------------
# Predict
# ------------------------------
if st.button("Predict Medical Cost"):
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Insurance Cost: ₹ {prediction[0]:,.2f}")
