# 🏥 Medical Cost Prediction App

A machine learning web application that predicts medical insurance charges based on user health and demographic details. Built using XGBoost regression and deployed with Streamlit.

---

## 🚀 Live Demo
👉 https://medical-cost-prediction-h4x9et8zub5mhcibxm7mmw.streamlit.app

---

## 📌 Project Overview

This project predicts the **insurance cost (charges)** of a person using features like age, BMI, smoking status, number of children, sex, and region.

It uses advanced machine learning techniques including:
- Feature engineering
- Log transformation of target variable
- XGBoost regression model
- Cross-validation for performance evaluation

---

## 📊 Dataset

- Source: Medical Insurance Dataset
- Features:
  - age
  - sex
  - bmi
  - children
  - smoker
  - region
- Target:
  - charges (medical cost)

---

## 🧠 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Streamlit
- Joblib

---

## ⚙️ ML Workflow

1. Data preprocessing
2. Feature engineering (BMI & age groups, interaction features)
3. Encoding categorical variables
4. Log transformation of target variable
5. Train-test split
6. Model training (XGBoost Regressor)
7. Evaluation using MAE, RMSE, R² score
8. Model saving using Joblib

---

## 📈 Model Performance

- MAE: ~2270
- RMSE: ~4678
- R² Score: ~0.85+

---

## 🖥️ How to Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/medical-cost-prediction.git

# Go to project folder
cd medical-cost-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
