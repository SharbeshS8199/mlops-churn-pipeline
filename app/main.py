"""Prediction API.   uvicorn app.main:app --port 8000"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn import __author__, __version__  # noqa: E402
from churn.serving import ChurnService  # noqa: E402

svc = ChurnService(os.environ.get("MODEL_PATH", str(ROOT / "models/champion/model.joblib")),
                   os.environ.get("PREDICTION_LOG", str(ROOT / "logs/predictions.jsonl")))
app = FastAPI(title="Churn prediction", version=__version__,
              contact={"name": __author__, "url": "https://github.com/SharbeshS8199"})

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100, examples=[3])
    MonthlyCharges: float = Field(ge=0, examples=[89.5])
    TotalCharges: float = Field(ge=0, examples=[268.5])
    SeniorCitizen: int = Field(0, ge=0, le=1)
    gender: Literal["Male", "Female"] = "Female"
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "Yes"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"] = "Electronic check"


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    b = svc.bundle
    return {"status": "ok", "model_version": b["version"], "trained_at": b["trained_at"], "metrics": b["metrics"]}


@app.post("/predict")
def predict(batch: Batch):
    try:
        return {"predictions": svc.predict([c.model_dump() for c in batch.customers])}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return svc.prometheus()
