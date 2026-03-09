def recommend_intervention(prob, features):

    attendance = features["attendance"]
    grade = features["average_grade"]
    delay = features["assignment_delay_days"]

    if prob > 0.75:
        return "Immediate faculty counseling and mentoring support"

    elif attendance < 60:
        return "Attendance monitoring and mentor intervention"

    elif grade < 60:
        return "Academic tutoring and study support program"

    elif delay > 7:
        return "Assignment deadline management coaching"

    else:
        return "Regular academic monitoring"