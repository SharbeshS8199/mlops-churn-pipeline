from churn.dashboard_data import compare_distribution, load_predictions, simulate, volume
from churn.data import synthetic


class FakeService:
    def __init__(self):
        self.calls = []

    def predict(self, recs, ts=None):
        self.calls.append((len(recs), ts))


def test_simulate_spreads_timestamps_over_hours():
    s = FakeService()
    assert simulate(s, n=500, hours=5, batch=50, seed=1) == 500
    ts = [t for _, t in s.calls]
    assert len(ts) == 10 and ts == sorted(ts) and 4 * 3600 < ts[-1] - ts[0] < 5 * 3600


def test_distribution_shares_sum_to_one():
    ref, cur = synthetic(1000, seed=1), synthetic(1000, seed=2, drift=True)
    for f in ("MonthlyCharges", "Contract"):
        d = compare_distribution(ref, cur, f)
        assert abs(d.groupby("data")["share"].sum() - 1).max() < 1e-9


def test_load_and_volume(tmp_path):
    import json
    import time
    p = tmp_path / "p.jsonl"
    now = time.time()
    p.write_text("\n".join(json.dumps({"ts": now - i * 600, "prediction": i / 20}) for i in range(20)))
    df = load_predictions(p)
    v = volume(df, "1h")
    assert v["requests"].sum() == 20 and "high_risk_share" in v
    assert load_predictions(tmp_path / "missing.jsonl").empty
