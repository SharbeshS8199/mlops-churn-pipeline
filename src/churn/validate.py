"""Data validation gate: schema, types, ranges, nulls, label balance. Run in CI before training and
on every batch of production data before it is used for retraining.

    python -m churn.validate            # exits 1 on failure
"""

from __future__ import annotations

import sys

import pandas as pd

from churn.data import CATEGORICAL, NUMERIC, TARGET, load

RANGES = {"tenure": (0, 100), "MonthlyCharges": (0, 500), "TotalCharges": (0, 50_000), "SeniorCitizen": (0, 1)}
ALLOWED = {"Contract": {"Month-to-month", "One year", "Two year"},
           "InternetService": {"DSL", "Fiber optic", "No"},
           TARGET: {"Yes", "No"}}


def validate(df: pd.DataFrame, require_target: bool = True, max_null: float = 0.01) -> list[str]:
    errors = []
    needed = NUMERIC + CATEGORICAL + ([TARGET] if require_target else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(df) < 100:
        errors.append(f"too few rows: {len(df)}")
    for c in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"{c} is not numeric")
            continue
        lo, hi = RANGES[c]
        bad = ((df[c] < lo) | (df[c] > hi)).mean()
        if bad > 0:
            errors.append(f"{c}: {bad:.2%} outside [{lo}, {hi}]")
    for c in needed:
        r = df[c].isna().mean()
        if r > max_null:
            errors.append(f"{c}: {r:.2%} nulls (max {max_null:.0%})")
    for c, allowed in ALLOWED.items():
        if c in df and not set(df[c].dropna().unique()) <= allowed:
            errors.append(f"{c}: unexpected values {set(df[c].dropna().unique()) - allowed}")
    if require_target:
        rate = (df[TARGET] == "Yes").mean()
        if not 0.02 <= rate <= 0.8:
            errors.append(f"churn rate {rate:.2%} looks wrong")
    return errors


def main() -> int:
    errs = validate(load())
    for e in errs:
        print("✗", e)
    print("data validation:", "FAILED" if errs else "passed")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
