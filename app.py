import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
from intervention_engine import recommend_intervention

# Page configuration
st.set_page_config(
    page_title="AI Dropout Prediction System",
    layout="wide"
)

# Load trained model
model = joblib.load("dropout_model.pkl")

# Title
st.title("🎓 AI-Powered Student Dropout Prediction System")

st.info(
"Privacy-Preserving Multimodal Behavioral Analytics System "
"for Early Dropout Risk Detection and Adaptive Intervention."
)

st.divider()

# -------------------------------
# STUDENT INDICATORS INPUT PANEL
# -------------------------------

st.header("📊 Student Behavioral Indicators")

st.markdown(
"Enter student academic, behavioral, and engagement indicators "
"to evaluate dropout risk."
)

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Academic indicators
with col1:

    st.subheader("📚 Academic Performance")

    attendance = st.number_input(
        "Attendance (%)",
        min_value=40,
        max_value=100,
        value=75
    )

    grade = st.number_input(
        "Average Grade",
        min_value=40,
        max_value=100,
        value=70
    )


# Engagement indicators
with col2:

    st.subheader("💻 Learning Engagement")

    logins = st.number_input(
        "LMS Logins per Month",
        min_value=0,
        max_value=40,
        value=10
    )

    forum = st.number_input(
        "Forum Participation",
        min_value=0,
        max_value=20,
        value=5
    )


# Behaviour indicators
with col3:

    st.subheader("⏳ Study Behaviour")

    delay = st.number_input(
        "Assignment Delay (days)",
        min_value=0,
        max_value=15,
        value=2
    )

    study = st.number_input(
        "Study Hours per Week",
        min_value=1,
        max_value=25,
        value=10
    )

st.divider()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

engagement_score = logins * 0.4 + forum * 0.6
academic_index = attendance * 0.5 + grade * 0.5
procrastination_index = delay * 0.7 - study * 0.3

features = np.array([[

attendance,
grade,
logins,
delay,
forum,
study,
engagement_score,
academic_index,
procrastination_index

]])

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("🔍 Predict Dropout Risk"):

    prob = model.predict_proba(features)[0][1]

    # Risk classification
    if prob < 0.3:
        risk = "Low"
        color = "green"
    elif prob < 0.6:
        risk = "Medium"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    st.divider()

    st.header("📈 Prediction Result")

    colA, colB = st.columns(2)

    with colA:

        st.metric(
            label="Dropout Probability",
            value=f"{prob:.2f}"
        )

        st.progress(prob)

        st.markdown(f"### Risk Level: :{color}[{risk}]")

    with colB:

        st.subheader("💡 Recommended Intervention")

        intervention = recommend_intervention(
            prob,
            {
                "attendance": attendance,
                "average_grade": grade,
                "assignment_delay_days": delay
            }
        )

        st.success(intervention)

    # -------------------------------
    # FEATURE IMPORTANCE VISUALIZATION
    # -------------------------------

    st.divider()

    st.header("🧠 Model Feature Importance")

    importance = model.feature_importances_

    feature_names = [
        "Attendance",
        "Average Grade",
        "LMS Logins",
        "Assignment Delay",
        "Forum Participation",
        "Study Hours",
        "Engagement Score",
        "Academic Index",
        "Procrastination Index"
    ]

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    chart = alt.Chart(df).mark_bar().encode(
        x="Importance",
        y=alt.Y("Feature", sort='-x')
    )

    st.altair_chart(chart, use_container_width=True)