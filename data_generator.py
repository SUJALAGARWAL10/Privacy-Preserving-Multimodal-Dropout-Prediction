import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "attendance": np.random.randint(40,100,n),
    "average_grade": np.random.randint(40,100,n),
    "lms_logins": np.random.randint(0,40,n),
    "assignment_delay_days": np.random.randint(0,15,n),
    "forum_participation": np.random.randint(0,20,n),
    "study_hours_per_week": np.random.randint(1,25,n)
})

# Feature Engineering
data["engagement_score"] = (
    data["lms_logins"] * 0.4 +
    data["forum_participation"] * 0.6
)

data["academic_index"] = (
    data["attendance"] * 0.5 +
    data["average_grade"] * 0.5
)

data["procrastination_index"] = (
    data["assignment_delay_days"] * 0.7 -
    data["study_hours_per_week"] * 0.3
)

# Dropout probability simulation
risk_score = (
    0.4*(100-data["attendance"]) +
    0.3*(100-data["average_grade"]) +
    0.2*data["assignment_delay_days"] +
    0.1*(30-data["lms_logins"])
)

data["dropout"] = (risk_score > np.percentile(risk_score,60)).astype(int)

data.to_csv("student_data.csv",index=False)

print("Dataset generated successfully")