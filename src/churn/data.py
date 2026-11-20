"""Customer churn data with the IBM Telco Customer Churn schema.

`load()` reads the real Kaggle file (WA_Fn-UseC_-Telco-Customer-Churn.csv) if present; otherwise
`synthetic()` generates data with the same columns so the whole pipeline runs anywhere.
`synthetic(..., drift=True)` simulates a later production period (price rise, more fibre users,
more electronic-check payers) for the drift monitor to catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "TechSupport", "StreamingTV", "Contract", "PaperlessBilling", "PaymentMethod"]
TARGET = "Churn"
REAL_FILE = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def synthetic(n: int = 7000, seed: int = 0, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda opts, p: rng.choice(opts, n, p=p)
    internet = pick(["DSL", "Fiber optic", "No"], [0.25, 0.6, 0.15] if drift else [0.34, 0.44, 0.22])
    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    payment = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                   [0.45, 0.15, 0.2, 0.2] if drift else [0.34, 0.23, 0.22, 0.21])
    tenure = np.clip(np.where(contract == "Month-to-month", rng.exponential(14, n), rng.uniform(5, 72, n)), 0, 72).astype(int)
    has_net = internet != "No"
    yes_no_net = lambda p: np.where(has_net, rng.choice(["Yes", "No"], n, p=[p, 1 - p]), "No internet service")
    online_sec, tech = yes_no_net(0.35), yes_no_net(0.35)
    streaming = yes_no_net(0.45)
    base = np.select([internet == "Fiber optic", internet == "DSL"], [80.0, 55.0], 20.0)
    monthly = np.round(base + (streaming == "Yes") * 10 + rng.normal(0, 6, n) + (12 if drift else 0), 2)
    phone = pick(["Yes", "No"], [0.9, 0.1])
    df = pd.DataFrame({
        "customerID": [f"C{seed:02d}{i:06d}" for i in range(n)],
        "gender": pick(["Male", "Female"], [0.5, 0.5]),
        "SeniorCitizen": (rng.random(n) < 0.16).astype(int),
        "Partner": pick(["Yes", "No"], [0.48, 0.52]),
        "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "Yes", rng.choice(["Yes", "No"], n), "No phone service"),
        "InternetService": internet, "OnlineSecurity": online_sec, "TechSupport": tech, "StreamingTV": streaming,
        "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"], [0.6, 0.4]),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, n), 2),
    })
    z = (-1.2 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year") - 0.04 * tenure
         + 0.7 * (internet == "Fiber optic") + 0.5 * (payment == "Electronic check") + 0.4 * df["SeniorCitizen"]
         - 0.5 * (online_sec == "Yes") - 0.4 * (tech == "Yes") + 0.015 * (monthly - 65) + rng.normal(0, 0.6, n))
    df[TARGET] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return df


def load(path: Path | None = None, n: int = 7000, seed: int = 0) -> pd.DataFrame:
    path = path or REAL_FILE
    if path.exists():
        df = pd.read_csv(path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)  # blank for new customers
        return df
    return synthetic(n, seed)
