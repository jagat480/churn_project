"""
app.py
------
FastAPI backend serving churn predictions from the trained pipeline.

Run locally with:
    uvicorn app:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI,
or POST to /predict with a JSON body matching the CustomerInput schema.
"""

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
)

# Allow a local frontend (e.g. Streamlit on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/churn_model.joblib")


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5,
                "TotalCharges": 480.0,
            }
        }


class PredictionOutput(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Churn prediction API is running. See /docs."}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    input_df = pd.DataFrame([customer.model_dump()])
    proba = model.predict_proba(input_df)[0, 1]
    prediction = "Yes" if proba >= 0.5 else "No"
    return PredictionOutput(churn_prediction=prediction, churn_probability=round(float(proba), 4))
