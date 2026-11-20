"""Framework-free serving core (the FastAPI app is a thin wrapper): load the champion, predict,
log every request for drift monitoring, and keep Prometheus-style counters."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

LATENCY_BUCKETS = (5, 10, 25, 50, 100, 250, 500)


class ChurnService:
    def __init__(self, model_path: str | Path = "models/champion/model.joblib",
                 log_path: str | Path = "logs/predictions.jsonl", threshold: float = 0.5):
        self.bundle = joblib.load(model_path)
        self.pipe, self.features = self.bundle["pipeline"], self.bundle["features"]
        self.log_path, self.threshold = Path(log_path), threshold
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.counts: Counter = Counter()
        self.latency_hist: Counter = Counter()
        self.latency_sum = 0.0
        self._lock = threading.Lock()

    def predict(self, records: list[dict], ts: float | None = None) -> list[dict]:
        """ts: override the log timestamp (used when replaying/backfilling simulated traffic)."""
        t0 = time.perf_counter()
        df = pd.DataFrame(records)
        missing = [f for f in self.features if f not in df]
        if missing:
            raise ValueError(f"missing features: {missing}")
        p = self.pipe.predict_proba(df[self.features])[:, 1]
        ms = (time.perf_counter() - t0) * 1000
        out = [{"churn_probability": round(float(x), 4), "will_churn": bool(x >= self.threshold),
                "model_version": self.bundle["version"]} for x in p]
        with self._lock:
            self.counts["requests"] += 1
            self.counts["predictions"] += len(out)
            self.counts["positive"] += sum(o["will_churn"] for o in out)
            self.latency_sum += ms
            for b in LATENCY_BUCKETS:
                if ms <= b:
                    self.latency_hist[b] += 1
            lines = [json.dumps({"ts": ts or time.time(), **{k: rec[k] for k in self.features},
                                 "prediction": o["churn_probability"]}) + "\n" for rec, o in zip(records, out)]
            if os.environ.get("LOG_TO_STDOUT"):
                # Cloud Run: stdout JSON -> Cloud Logging -> log sink to GCS (see deploy/GCP.md)
                sys.stdout.writelines(lines)
                sys.stdout.flush()
            else:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.writelines(lines)
        return out

    def prometheus(self) -> str:
        c = self.counts
        lines = ["# TYPE churn_requests_total counter", f"churn_requests_total {c['requests']}",
                 "# TYPE churn_predictions_total counter", f"churn_predictions_total {c['predictions']}",
                 "# TYPE churn_positive_predictions_total counter", f"churn_positive_predictions_total {c['positive']}",
                 "# TYPE churn_request_latency_ms histogram"]
        lines += [f'churn_request_latency_ms_bucket{{le="{b}"}} {self.latency_hist[b]}' for b in LATENCY_BUCKETS]
        lines += [f'churn_request_latency_ms_bucket{{le="+Inf"}} {c["requests"]}',
                  f"churn_request_latency_ms_sum {self.latency_sum:.3f}", f"churn_request_latency_ms_count {c['requests']}"]
        return "\n".join(lines) + "\n"
