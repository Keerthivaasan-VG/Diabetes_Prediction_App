import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open("diabetes_model.pkl", "rb") as f:
    Logr = pickle.load(f)

st.title("Diabetes Prediction App")
st.write("Enter the following details to predict diabetes:")

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

