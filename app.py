import os
import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model
# ==============================
model = joblib.load("medical_cost_model.pkl")

st.title("🏥 Medical Insurance Cost Prediction")

st.write("Enter patient details")

# ==============================
# Inputs
# ==============================
age = st.number_input("Age", 18, 100, 25)

sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 10, 0)

smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ==============================
# Encoding (must match training)
# ==============================
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}
region = region_map[region]

# ==============================
# FINAL INPUT (ONLY ORIGINAL FEATURES)
# ==============================
input_data = pd.DataFrame([[
    age,
    sex,
    bmi,
    children,
    smoker,
    region
]], columns=[
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region"
])

# ==============================
# Debug check
# ==============================
st.write("Model loaded:", os.path.exists("medical_cost_model.pkl"))

# ==============================
# Prediction
# ==============================
if st.button("Predict Cost"):

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Insurance Cost: ₹ {prediction[0]:,.2f}")
