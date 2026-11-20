"""Drift monitoring: compare recent production inputs (logs/predictions.jsonl) with the training
reference sample.

    python -m churn.drift --window 2000        # exit code 2 if drift is detected (triggers retraining in CI)

Uses Evidently's DataDriftPreset for the HTML report when installed; the decision itself is made
with PSI + KS/chi-square computed here, so it's transparent, testable and version-independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from churn.data import CATEGORICAL, NUMERIC

PSI_WARN, PSI_ALERT = 0.1, 0.25          # common industry rule of thumb


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def detect(ref: pd.DataFrame, cur: pd.DataFrame, share_threshold: float = 0.3) -> dict:
    rows = []
    for col in NUMERIC:
        psi = psi_numeric(ref[col].to_numpy(float), cur[col].to_numpy(float))
        p = float(stats.ks_2samp(ref[col], cur[col]).pvalue)
        rows.append({"feature": col, "type": "numeric", "psi": round(psi, 4), "p_value": p,
                     "drift": psi >= PSI_ALERT or (psi >= PSI_WARN and p < 0.01)})
    for col in CATEGORICAL:
        psi = psi_categorical(ref[col].astype(str), cur[col].astype(str))
        cats = sorted(set(ref[col].astype(str)) | set(cur[col].astype(str)))
        table = np.array([ref[col].astype(str).value_counts().reindex(cats, fill_value=0),
                          cur[col].astype(str).value_counts().reindex(cats, fill_value=0)])
        table = table[:, table.sum(0) > 0]
        p = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else 1.0
        rows.append({"feature": col, "type": "categorical", "psi": round(psi, 4), "p_value": p,
                     "drift": psi >= PSI_ALERT or (psi >= PSI_WARN and p < 0.01)})
    share = sum(r["drift"] for r in rows) / len(rows)
    return {"n_reference": len(ref), "n_current": len(cur), "drifted_share": round(share, 3),
            "dataset_drift": share >= share_threshold or any(r["drift"] and r["feature"] in ("MonthlyCharges", "Contract")
                                                              for r in rows),
            "features": sorted(rows, key=lambda r: -r["psi"])}


def evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> bool:
    try:                                   # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        Report([DataDriftPreset()]).run(cur, ref).save_html(str(path))
        return True
    except ImportError:
        pass
    try:                                   # Evidently 0.4.x
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        r = Report(metrics=[DataDriftPreset()])
        r.run(reference_data=ref, current_data=cur)
        r.save_html(str(path))
        return True
    except ImportError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="models/champion/reference.csv")
    ap.add_argument("--log", default="logs/predictions.jsonl")
    ap.add_argument("--window", type=int, default=2000, help="most recent N predictions")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    ref = pd.read_csv(args.reference)
    cur = pd.read_json(args.log, lines=True).tail(args.window)
    res = detect(ref, cur)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "drift.json").write_text(json.dumps(res, indent=2))
    md = [f"**Drifted features:** {res['drifted_share']:.0%} of {len(res['features'])} "
          f"(window: last {res['n_current']} predictions vs {res['n_reference']} training rows)", "",
          "| feature | PSI | p-value | drift |", "|---|---|---|---|"]
    md += [f"| {r['feature']} | {r['psi']:.3f} | {r['p_value']:.2g} | {'⚠️' if r['drift'] else ''} |" for r in res["features"]]
    (out / "drift.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    html = evidently_html(ref, cur[ref.columns], out / "drift_report.html")
    print(f"drifted features: {res['drifted_share']:.0%}  dataset drift: {res['dataset_drift']}")
    for r in res["features"][:6]:
        print(f"  {r['feature']:18s} PSI {r['psi']:.3f}  p={r['p_value']:.2g}  {'DRIFT' if r['drift'] else ''}")
    print("Evidently report:", out / "drift_report.html" if html else "(install evidently for the HTML report)")
    return 2 if res["dataset_drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
