import streamlit as st
import joblib
import numpy as np

# Load the pre-trained SVM model
model = joblib.load('svm_model.pkl')  # Ensure svm_model.pkl is uploaded

# App title
st.title("Product Purchase Prediction")
st.write("This app predicts whether a user will purchase a product based on their details.")

# Input fields
st.sidebar.header("User Input Parameters")
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
age = st.sidebar.slider("Age", 18, 60, 25)
estimated_salary = st.sidebar.slider("Estimated Salary", 15000, 150000, 50000, step=500)

# Convert Gender to numeric (Male=1, Female=0)
gender_numeric = 1 if gender == "Male" else 0

# Prepare input for prediction
input_data = np.array([[gender_numeric, age, estimated_salary]])

# Prediction button
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("The user is likely to purchase the product.")
    else:
        st.error("The user is unlikely to purchase the product.")
