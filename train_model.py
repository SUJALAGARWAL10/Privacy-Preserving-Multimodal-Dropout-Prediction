import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("student_data.csv")

X = data.drop("dropout",axis=1)
y = data["dropout"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))

joblib.dump(model,"dropout_model.pkl")

print("Model trained and saved")