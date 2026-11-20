"""MLOps control room.   streamlit run dashboard/app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "dashboard")]
import ui  # noqa: E402
from churn import train as train_mod  # noqa: E402
from churn.dashboard_data import compare_distribution, load_predictions, load_runs, simulate, volume  # noqa: E402
from churn.drift import PSI_ALERT, PSI_WARN, detect  # noqa: E402

ui.page("Churn Model · MLOps Control Room", "Tracked training · quality gate · live service · drift monitoring", "🛰️")
CHAMP, LOG = ROOT / "models" / "champion", ROOT / "logs" / "predictions.jsonl"
params = yaml.safe_load(open(ROOT / "params.yaml"))


def retrain():
    import os

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return train_mod.main(str(ROOT / "params.yaml"))
    finally:
        os.chdir(cwd)


if not (CHAMP / "model.joblib").exists():
    ui.empty_state("No model in production yet", "Train once: the run is tracked, checked by the quality gate and "
                   "promoted to champion if it passes.", "python -m churn.train")
    if st.button("Train and promote", type="primary"):
        with st.spinner("Training…"):
            retrain()
        st.rerun()
    ui.footer()
    st.stop()


@st.cache_resource
def service():
    from churn.serving import ChurnService

    return ChurnService(CHAMP / "model.joblib", LOG)


svc = service()
with st.sidebar:
    st.subheader("Simulate production")
    st.caption("Replays 2,000 synthetic customers through the live model, spread over the last 8 hours.")
    if st.button("➕ Normal traffic", use_container_width=True):
        simulate(svc, drift=False)
        st.rerun()
    if st.button("⚠️ Drifted traffic (price rise + mix shift)", use_container_width=True):
        simulate(svc, drift=True)
        st.rerun()
    if st.button("🗑️ Clear prediction log", use_container_width=True):
        LOG.unlink(missing_ok=True)
        st.rerun()
    st.divider()
    st.subheader("Pipeline")
    if st.button("🔁 Retrain with quality gate", use_container_width=True):
        with st.spinner("Training candidate…"):
            code = retrain()
        st.session_state["last_train"] = "promoted to champion" if code == 0 else "blocked by the quality gate"
        st.cache_resource.clear()
        st.rerun()
    if "last_train" in st.session_state:
        st.info(f"Last retrain: {st.session_state['last_train']}")
    window = st.slider("Drift window (latest predictions)", 500, 10000, 2000, 500)

champ = json.loads((CHAMP / "metrics.json").read_text())
bundle_meta = svc.bundle
preds = load_predictions(LOG)
ref = pd.read_csv(CHAMP / "reference.csv")
drift = detect(ref, preds.tail(window)) if len(preds) >= 200 else None

status = ("No traffic yet", ui.MUTED) if drift is None else \
    (("Drift detected", ui.BAD) if drift["dataset_drift"] else ("Stable", ui.GOOD))
ui.kpis([
    ("Champion", champ["run_id"][:8], f"trained {bundle_meta['trained_at'].replace('T', ' ')}"),
    ("ROC-AUC", f"{champ['roc_auc']:.3f}", f"gate ≥ {params['gate']['min_roc_auc']}"),
    ("PR-AUC", f"{champ['pr_auc']:.3f}", f"gate ≥ {params['gate']['min_pr_auc']}"),
    ("Predictions logged", f"{len(preds):,}", f"{svc.counts['requests']} requests this session"),
    ("Data drift", ui.pill(status[0], status[1]), "" if drift is None else f"{drift['drifted_share']:.0%} of features drifted"),
])
st.write("")
t_drift, t_live, t_reg = st.tabs(["🌊 Drift", "📈 Live service", "🗂️ Model registry & runs"])

with t_drift:
    if drift is None:
        ui.empty_state("Not enough production data", "Use the sidebar to send simulated traffic through the model, "
                       "then compare it with the training data here.")
    else:
        feats = pd.DataFrame(drift["features"])
        feats["status"] = feats["drift"].map({True: "drift", False: "ok"})
        c1, c2 = st.columns([2, 3])
        f = px.bar(feats.sort_values("psi"), x="psi", y="feature", orientation="h", color="status",
                   color_discrete_map={"drift": ui.BAD, "ok": ui.GOOD},
                   title=f"Population Stability Index (last {len(preds.tail(window)):,} vs training)")
        f.add_vline(x=PSI_WARN, line_dash="dot", line_color=ui.WARN)
        f.add_vline(x=PSI_ALERT, line_dash="dash", line_color=ui.BAD)
        c1.plotly_chart(ui.style_fig(f, 460), use_container_width=True)
        with c2:
            feature = st.selectbox("Compare a feature", feats["feature"].tolist())
            dist = compare_distribution(ref, preds.tail(window), feature)
            fd = px.bar(dist, x="bin", y="share", color="data", barmode="group",
                        title=f"{feature}: training vs production", color_discrete_sequence=[ui.MUTED, ui.COLORS[0]])
            st.plotly_chart(ui.style_fig(fd, 330), use_container_width=True)
            row = feats.set_index("feature").loc[feature]
            st.markdown(f"PSI **{row['psi']:.3f}** · test p-value **{row['p_value']:.2g}** · "
                        + ui.pill(row["status"].upper(), ui.BAD if row["drift"] else ui.GOOD), unsafe_allow_html=True)
        st.caption(f"Rule: drift if PSI ≥ {PSI_ALERT}, or PSI ≥ {PSI_WARN} with p < 0.01 (KS for numeric, χ² for "
                   "categorical). In GitHub Actions the same check runs daily; on drift it opens an issue and triggers retrain.yml.")

with t_live:
    if preds.empty:
        ui.empty_state("No predictions logged yet", "Send traffic from the sidebar, or call the API: "
                       "`uvicorn app.main:app` → POST /predict.")
    else:
        v = volume(preds)
        c1, c2 = st.columns(2)
        c1.plotly_chart(ui.style_fig(px.bar(v, x="time", y="requests", title="Predictions per 30 min"), 300),
                        use_container_width=True)
        c2.plotly_chart(ui.style_fig(px.line(v, x="time", y="mean_score", markers=True,
                                             title="Mean predicted churn probability"), 300), use_container_width=True)
        fh = px.histogram(preds.tail(window), x="prediction", nbins=40, title="Score distribution (latest window)")
        fh.add_vline(x=0.5, line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(fh, 280), use_container_width=True)
        st.code(svc.prometheus(), language="text")
        st.caption("Prometheus metrics, as exposed at GET /metrics.")

with t_reg:
    runs = load_runs(ROOT / "runs")
    if runs.empty:
        st.caption("No tracked runs found in ./runs (or MLflow).")
    else:
        runs["gate"] = runs["gate_passed"].map({True: "passed", False: "blocked"}).fillna("unknown")
        fr = px.scatter(runs, x="started", y="roc_auc", color="gate", size="pr_auc", hover_data=["run_id"],
                        color_discrete_map={"passed": ui.GOOD, "blocked": ui.BAD, "unknown": ui.MUTED},
                        title="Every training run, coloured by quality-gate result")
        fr.add_hline(y=params["gate"]["min_roc_auc"], line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(fr, 320), use_container_width=True)
        st.dataframe(runs[["started", "run_id", "model_type", "roc_auc", "pr_auc", "f1_at_0.5", "brier", "gate"]]
                     .sort_values("started", ascending=False), hide_index=True, use_container_width=True)
    st.markdown("**Quality gate** (from `params.yaml`)")
    st.json(params["gate"])

ui.footer("CI/CD: GitHub Actions → Artifact Registry → Cloud Run · see deploy/GCP.md")
