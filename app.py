import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("dropout_model.pkl")

st.title("🎓 Student Dropout Risk Predictor")

st.write("Enter student details to predict dropout risk")

# Input fields
attendance = st.slider("Attendance (%)", 0, 100, 75)
average_grade = st.slider("Average Grade", 0, 100, 70)
lms_logins = st.slider("LMS Logins per month", 0, 50, 15)
study_hours = st.slider("Study Hours per week", 0, 40, 10)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[attendance, average_grade, lms_logins, study_hours]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Dropout")
    else:
        st.success("✅ Low Risk of Dropout")