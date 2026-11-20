"""Train -> evaluate -> quality gate -> register/promote.

    python -m churn.train                  # exit code 1 if the candidate fails the gate (used by CI)

Artifacts:
  models/candidate/   model.joblib, metrics.json, reference.csv (training feature sample for drift)
  models/champion/    the model being served; replaced only when the candidate passes the gate
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import __version__
from churn.data import CATEGORICAL, NUMERIC, TARGET, load
from churn.tracking import tracker
from churn.validate import validate

MODELS = Path("models")


def build_pipeline(cfg: dict) -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    m = cfg["model"]
    if m["type"] == "logreg":
        clf = LogisticRegression(max_iter=2000)
    else:
        clf = HistGradientBoostingClassifier(learning_rate=m["learning_rate"], max_iter=m["max_iter"],
                                             max_leaf_nodes=m["max_leaf_nodes"], l2_regularization=m["l2_regularization"],
                                             early_stopping=True, random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
            "f1_at_0.5": round(float(f1_score(y, p >= 0.5)), 4), "brier": round(float(brier_score_loss(y, p)), 4)}


def gate(m: dict, champion: dict | None, g: dict) -> list[str]:
    fails = []
    if m["roc_auc"] < g["min_roc_auc"]:
        fails.append(f"roc_auc {m['roc_auc']} < {g['min_roc_auc']}")
    if m["pr_auc"] < g["min_pr_auc"]:
        fails.append(f"pr_auc {m['pr_auc']} < {g['min_pr_auc']}")
    if champion and m["roc_auc"] < champion["roc_auc"] - g["max_regression"]:
        fails.append(f"roc_auc {m['roc_auc']} regresses vs champion {champion['roc_auc']}")
    return fails


def main(params: str = "params.yaml", data: pd.DataFrame | None = None, models_dir: Path = MODELS) -> int:
    cfg = yaml.safe_load(open(params))
    df = data if data is not None else load(n=cfg["data"]["rows"], seed=cfg["data"]["seed"])
    errs = validate(df)
    if errs:
        print("data validation failed:", errs)
        return 1
    X, y = df[NUMERIC + CATEGORICAL], (df[TARGET] == "Yes").astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=42)

    tr = tracker()
    run_id = tr.start("churn")
    tr.log_params({"model." + k: v for k, v in cfg["model"].items()} | {"rows": len(df), "code_version": __version__})
    t0 = time.perf_counter()
    pipe = build_pipeline(cfg).fit(Xtr, ytr)
    m = metrics(yte, pipe.predict_proba(Xte)[:, 1]) | {"train_seconds": round(time.perf_counter() - t0, 2)}
    tr.log_metrics(m)

    cand = models_dir / "candidate"
    cand.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipe, "version": run_id, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "features": NUMERIC + CATEGORICAL, "metrics": m}
    joblib.dump(bundle, cand / "model.joblib")
    (cand / "metrics.json").write_text(json.dumps(m | {"run_id": run_id}, indent=2))
    Xtr.sample(min(2000, len(Xtr)), random_state=0).to_csv(cand / "reference.csv", index=False)
    tr.log_artifact(cand / "metrics.json")
    tr.log_model(pipe, "model")

    champ_file = models_dir / "champion" / "metrics.json"
    champion = json.loads(champ_file.read_text()) if champ_file.exists() else None
    fails = gate(m, champion, cfg["gate"])
    tr.log_params({"gate_passed": not fails})
    tr.end()
    print(json.dumps(m, indent=2))
    if fails:
        print("QUALITY GATE FAILED:", "; ".join(fails))
        return 1
    if (models_dir / "champion").exists():
        shutil.rmtree(models_dir / "champion")
    shutil.copytree(cand, models_dir / "champion")
    print(f"gate passed -> promoted run {run_id} to champion"
          + (f" (previous roc_auc {champion['roc_auc']})" if champion else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
