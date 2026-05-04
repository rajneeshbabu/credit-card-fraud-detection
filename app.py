from fastapi import FastAPI
import joblib
import json
import pandas as pd

app = FastAPI(title="Credit Card Fraud Detection API")

# Load artifacts
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/model_info.json") as f:
    info = json.load(f)

FEATURES = info["features"]
THRESHOLD = info["best_threshold"]


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        X = pd.DataFrame([data], columns=FEATURES)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob >= THRESHOLD)

        return {
            "fraud_probability": float(prob),
            "prediction": pred,  # 1 = fraud, 0 = legit
            "threshold": THRESHOLD,
        }

    except Exception as e:
        return {"error": str(e)}
