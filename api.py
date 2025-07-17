from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import shap
import numpy as np

# Load the saved model
model = joblib.load("rf_model.pkl")
explainer = shap.TreeExplainer(model)

# Define the threshold
THRESHOLD = 0.22

# Create the FastAPI app
app = FastAPI(title="Fraud Detection API")

# Define the input schema
class Transaction(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to NumPy array and reshape
    X = np.array(transaction.features).reshape(1, -1)

    # Get probability of fraud (class 1)
    proba = model.predict_proba(X)[0][1]

    # Apply threshold
    prediction = int(proba >= THRESHOLD)
    shap_values = explainer.shap_values(X)  # class 1 shap values
    shap_list = shap_values[0].tolist()

    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": prediction,
        "shap_values": shap_list
    }
