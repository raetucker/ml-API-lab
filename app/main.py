from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

class PatientData(BaseModel):
    age: float = Field(..., description="Normalized age")
    sex: float = Field(..., description="Normalized sex")
    bmi: float = Field(..., description="Body mass index")
    bp: float  = Field(..., description="Blood pressure")
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 0.05,
                    "sex": 0.05,
                    "bmi": 0.06,
                    "bp": 0.02,
                    "s1": -0.04,
                    "s2": -0.04,
                    "s3": -0.02,
                    "s4": -0.01,
                    "s5": 0.01,
                    "s6": 0.02
                }
            ]
        }
    }

app = FastAPI(
    title="Diabetes Progression Predictor",
    description="Predicts diabetes progression score from physiological features",
    version="1.0.0"
)

model_path = os.path.join("models", "diabetes_model.pkl")
if not os.path.exists(model_path):
    raise RuntimeError("Model not found. Run `python train_model.py` first.")

model = joblib.load(model_path)

def get_interpretation(score: float):
    if score < 100:
        return "Below average progression"
    elif score < 150:
        return "Average progression"
    else:
        return "Above average progression"

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "diabetes_progression_v1"}

@app.post("/predict")
def predict_progression(patient: PatientData):
    try:
        features = np.array([[
            patient.age, patient.sex, patient.bmi, patient.bp,
            patient.s1, patient.s2, patient.s3, patient.s4,
            patient.s5, patient.s6
        ]])
        prediction = float(model.predict(features)[0])
        return {
            "predicted_progression_score": round(prediction, 2),
            "interpretation": get_interpretation(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
