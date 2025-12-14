import streamlit as st
import pickle
import numpy as np


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩸",
    layout="centered"
)

# Load the trained logistic regression model
with open("diabetes_model.pkl", "rb") as f:
    Logr = pickle.load(f)

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>💉 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-assisted Diabetes Prediction</p>", unsafe_allow_html=True)
st.markdown("---")


# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=25)
mass = st.number_input("Mass (BMI)", min_value=1.0, max_value=100.0, value=25.0)
insu = st.number_input("Insulin Level", min_value=0, max_value=900, value=100)
plas = st.number_input("Plasma Glucose", min_value=0, max_value=200, value=120)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])
    prediction = Logr.predict(input_data)[0]
    
    if prediction == "tested_positive":
        st.error("The person may have diabetes.")
    else:
        st.success("The person is likely healthy.")

# Optional: Show model accuracy (from your training)
if st.checkbox("Show Model Accuracy"):
    st.write("Model Accuracy on training data: 76.69%")

