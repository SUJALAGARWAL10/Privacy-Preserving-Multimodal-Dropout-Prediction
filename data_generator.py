import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "attendance": np.random.randint(40, 100, n),
    "average_grade": np.random.randint(40, 100, n),
    "lms_logins": np.random.randint(0, 30, n),
    "assignment_delay_days": np.random.randint(0, 10, n)
})

# Simple dropout logic (for synthetic label)
data["dropout"] = (
    (data["attendance"] < 60) &
    (data["average_grade"] < 60)
).astype(int)

data.to_csv("student_data.csv", index=False)

print("Dataset Created")