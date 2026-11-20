"""Data helpers for the MLOps dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from churn.data import CATEGORICAL, NUMERIC, synthetic


def load_runs(root: str | Path = "runs") -> pd.DataFrame:
    """Run history from the JSON tracker (or MLflow when it is installed and has runs)."""
    rows = []
    for f in Path(root).glob("*/*/run.json"):
        r = json.loads(f.read_text())
        rows.append({"run_id": r["run_id"], "started": pd.to_datetime(r["start"], unit="s"), **r["metrics"],
                     "gate_passed": r["params"].get("gate_passed"), "model_type": r["params"].get("model.type")})
    try:
        import mlflow

        mr = mlflow.search_runs(experiment_names=["churn"])
        for _, r in mr.iterrows():
            rows.append({"run_id": r["run_id"], "started": r["start_time"],
                         "roc_auc": r.get("metrics.roc_auc"), "pr_auc": r.get("metrics.pr_auc"),
                         "f1_at_0.5": r.get("metrics.f1_at_0.5"), "brier": r.get("metrics.brier"),
                         "gate_passed": r.get("params.gate_passed") == "True", "model_type": r.get("params.model.type")})
    except Exception:
        pass
    df = pd.DataFrame(rows)
    return df.sort_values("started").reset_index(drop=True) if len(df) else df


def load_predictions(log: str | Path) -> pd.DataFrame:
    p = Path(log)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_json(p, lines=True)
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df


def volume(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    g = df.set_index("time").resample(freq).agg(requests=("prediction", "size"), mean_score=("prediction", "mean"))
    g["high_risk_share"] = df.set_index("time")["prediction"].ge(0.5).resample(freq).mean()
    return g.reset_index()


def compare_distribution(ref: pd.DataFrame, cur: pd.DataFrame, feature: str, bins: int = 20) -> pd.DataFrame:
    """Long table: share of rows per bin/category for the training reference vs recent production."""
    if feature in NUMERIC:
        edges = np.unique(np.quantile(pd.concat([ref[feature], cur[feature]]), np.linspace(0, 1, bins + 1)))
        rows = []
        for name, d in (("training reference", ref), ("recent production", cur)):
            h, _ = np.histogram(d[feature], edges)
            rows += [{"bin": f"{edges[i]:.0f}–{edges[i + 1]:.0f}", "order": i, "share": c / max(1, len(d)), "data": name}
                     for i, c in enumerate(h)]
        return pd.DataFrame(rows).sort_values("order")
    rows = []
    for name, d in (("training reference", ref), ("recent production", cur)):
        vc = d[feature].astype(str).value_counts(normalize=True)
        rows += [{"bin": k, "order": 0, "share": v, "data": name} for k, v in vc.items()]
    return pd.DataFrame(rows)


def simulate(service, n: int = 2000, drift: bool = False, hours: float = 8.0, batch: int = 50, seed: int | None = None) -> int:
    """Replay synthetic traffic through the service, spread over the last `hours` (backfilled timestamps)."""
    seed = int(time.time()) % 10_000 if seed is None else seed
    df = synthetic(n, seed=seed, drift=drift)[NUMERIC + CATEGORICAL]
    recs = df.to_dict(orient="records")
    nb = max(1, len(recs) // batch)
    now = time.time()
    for i in range(nb):
        service.predict(recs[i * batch:(i + 1) * batch], ts=now - (nb - i) * hours * 3600 / nb)
    return len(recs)
