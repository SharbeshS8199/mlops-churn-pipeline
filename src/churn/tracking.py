"""Experiment tracking: MLflow when installed (and MLFLOW_TRACKING_URI or ./mlruns), otherwise a
tiny JSON tracker with the same calls so the pipeline still records every run."""

from __future__ import annotations

import importlib.util
import json
import time
import uuid
from pathlib import Path

HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


class LocalTracker:
    def __init__(self, root: str = "runs"):
        self.root = Path(root)

    def start(self, experiment: str) -> str:
        self.run_id = uuid.uuid4().hex[:12]
        self.dir = self.root / experiment / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record = {"run_id": self.run_id, "experiment": experiment, "start": time.time(), "params": {}, "metrics": {}}
        return self.run_id

    def log_params(self, p: dict) -> None:
        self.record["params"].update(p)

    def log_metrics(self, m: dict) -> None:
        self.record["metrics"].update(m)

    def log_artifact(self, path: str | Path) -> None:
        self.record.setdefault("artifacts", []).append(str(path))

    def log_model(self, model, name: str) -> None:
        import joblib

        joblib.dump(model, self.dir / f"{name}.joblib")

    def end(self) -> None:
        self.record["end"] = time.time()
        (self.dir / "run.json").write_text(json.dumps(self.record, indent=2, default=str))


class MlflowTracker:
    def start(self, experiment: str) -> str:
        import mlflow

        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run()
        return self.run.info.run_id

    def log_params(self, p: dict) -> None:
        import mlflow

        mlflow.log_params(p)

    def log_metrics(self, m: dict) -> None:
        import mlflow

        mlflow.log_metrics(m)

    def log_artifact(self, path: str | Path) -> None:
        import mlflow

        mlflow.log_artifact(str(path))

    def log_model(self, model, name: str) -> None:
        import mlflow.sklearn

        mlflow.sklearn.log_model(model, name=name, registered_model_name="churn-model")

    def end(self) -> None:
        import mlflow

        mlflow.end_run()


def tracker():
    return MlflowTracker() if HAS_MLFLOW else LocalTracker()
