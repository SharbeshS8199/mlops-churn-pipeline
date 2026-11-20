import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from churn import train as train_mod
from churn.data import CATEGORICAL, NUMERIC, synthetic
from churn.drift import detect, psi_categorical, psi_numeric
from churn.serving import ChurnService
from churn.validate import validate

ROOT = Path(__file__).resolve().parents[1]


def test_validation_catches_bad_data():
    df = synthetic(1000)
    assert validate(df) == []
    bad = df.copy()
    bad.loc[:50, "tenure"] = -5
    bad.loc[:100, "MonthlyCharges"] = np.nan
    bad.loc[0, "Contract"] = "Weekly"
    errs = validate(bad)
    assert any("tenure" in e for e in errs) and any("nulls" in e for e in errs) and any("Contract" in e for e in errs)
    assert "missing columns" in validate(df.drop(columns=["tenure"]))[0]


def test_gate_rules():
    g = {"min_roc_auc": 0.75, "min_pr_auc": 0.5, "max_regression": 0.01}
    assert train_mod.gate({"roc_auc": 0.8, "pr_auc": 0.6}, None, g) == []
    assert train_mod.gate({"roc_auc": 0.7, "pr_auc": 0.6}, None, g)
    assert train_mod.gate({"roc_auc": 0.80, "pr_auc": 0.6}, {"roc_auc": 0.83}, g)      # regression vs champion
    assert train_mod.gate({"roc_auc": 0.825, "pr_auc": 0.6}, {"roc_auc": 0.83}, g) == []


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    assert train_mod.main(str(ROOT / "params.yaml"), synthetic(4000, seed=3), d) == 0
    return d


def test_train_promotes_and_blocks_worse_model(trained, tmp_path):
    champ = json.loads((trained / "champion" / "metrics.json").read_text())
    assert champ["roc_auc"] >= 0.75 and (trained / "champion" / "reference.csv").exists()
    cfg = yaml.safe_load(open(ROOT / "params.yaml"))
    cfg["model"].update(max_iter=1, learning_rate=0.001)            # a deliberately bad candidate
    cfg["gate"]["min_roc_auc"] = 0.0
    cfg["gate"]["min_pr_auc"] = 0.0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert train_mod.main(str(p), synthetic(4000, seed=3), trained) == 1  # regression vs champion -> blocked
    assert json.loads((trained / "champion" / "metrics.json").read_text())["run_id"] == champ["run_id"]


def test_service_predicts_logs_and_exposes_metrics(trained, tmp_path):
    svc = ChurnService(trained / "champion" / "model.joblib", tmp_path / "p.jsonl")
    recs = synthetic(20, seed=5)[NUMERIC + CATEGORICAL].to_dict(orient="records")
    out = svc.predict(recs)
    assert len(out) == 20 and all(0 <= o["churn_probability"] <= 1 for o in out)
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 20
    prom = svc.prometheus()
    assert "churn_predictions_total 20" in prom and 'le="+Inf"} 1' in prom
    with pytest.raises(ValueError):
        svc.predict([{"tenure": 1}])


def test_psi_properties():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi_numeric(a, rng.normal(0, 1, 5000)) < 0.02
    assert psi_numeric(a, rng.normal(1, 1, 5000)) > 0.25
    import pandas as pd
    s = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(s, s) < 1e-6
    assert psi_categorical(s, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.25


def test_drift_detector_separates_normal_from_shifted():
    ref = synthetic(3000, seed=1)
    assert not detect(ref, synthetic(2000, seed=2))["dataset_drift"]
    res = detect(ref, synthetic(2000, seed=2, drift=True))
    assert res["dataset_drift"] and res["features"][0]["feature"] == "MonthlyCharges"
