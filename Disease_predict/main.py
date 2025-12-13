from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import statistics

app = FastAPI(title="Disease Prediction API")

# Load models
svm = joblib.load("svm_model.pkl")
nb = joblib.load("nb_model.pkl")
rb = joblib.load("rb_model.pkl")
encoder = joblib.load("encoder.pkl")
symptoms_list = joblib.load("symptoms.pkl")

# create symptom index
symptom_index = {sym.replace("_", " ").title(): i for i, sym in enumerate(symptoms_list)}

class SymptomsInput(BaseModel):
    symptoms: str

@app.post("/predict")
def predict_disease(data: SymptomsInput):
    input_data = [0] * len(symptom_index)
    invalid_symptoms = []

    for symptom in data.symptoms.split(","):
        symptom = symptom.strip().title()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    svm_pred = encoder.classes_[svm.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb.predict(input_data)[0]]
    rb_pred = encoder.classes_[rb.predict(input_data)[0]]

    final_pred = max(
        set([svm_pred, nb_pred, rb_pred]),
        key=[svm_pred, nb_pred, rb_pred].count
    )
    return {
        "svm_prediction": svm_pred,
        "nb_prediction": nb_pred,
        "random_forest_prediction": rb_pred,
        "final_prediction": final_pred
    }