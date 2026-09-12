from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "titanic_best_model.joblib"
model = joblib.load(MODEL_PATH)

example_passenger = pd.DataFrame([{
    "PassengerId": 9999,
    "Pclass": 1,
    "Name": "Example, Mrs. Jane",
    "Sex": "female",
    "Age": 30,
    "SibSp": 0,
    "Parch": 1,
    "Ticket": "TEST123",
    "Fare": 80.0,
    "Cabin": "C85",
    "Embarked": "S",
}])

prediction = int(model.predict(example_passenger)[0])
probability = float(model.predict_proba(example_passenger)[0, 1])

print("Predicted survival:", prediction)
print("Survival probability:", f"{probability:.2%}")
