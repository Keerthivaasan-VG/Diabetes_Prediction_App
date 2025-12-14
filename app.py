import streamlit as st
import pickle
import numpy as np

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩸",
    layout="centered"
)

# ================== LOAD MODEL ==================
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>🩸 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-Assisted Diabetes Risk Prediction</p>", unsafe_allow_html=True)
st.markdown("---")

# ================== INPUT FIELDS ==================
age = st.number_input("Age (years)", min_value=1, max_value=100, value=25)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=100)
glucose = st.number_input("Plasma Glucose Level (mg/dL)", min_value=50, max_value=200, value=120)

# ================== PREDICTION ==================
if st.button("🔍 Predict"):
    # Order MUST match training data
    input_data = np.array([[age, bmi, insulin, glucose]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The person is likely **Diabetic**.")
    else:
        st.success("✅ The person is likely **Non-Diabetic**.")

# ================== OPTIONAL INFO ==================
if st.checkbox("📊 Show Model Accuracy"):
    st.info("Model Accuracy on Training Data: **76.69%**")
