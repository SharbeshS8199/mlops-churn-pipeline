export PYTHONPATH := src
.PHONY: install validate train serve traffic drift-demo test up

install:  ; pip install -e '.[all]'
validate: ; python -m churn.validate
train:    ; python -m churn.train
serve:    ; uvicorn app.main:app --port 8000 --reload
traffic:  ; python -m churn.simulate_traffic --n 2000
test:     ; pytest -q
up:       ; python -m churn.train && docker compose up --build

# Full loop locally: train, normal traffic (no drift), shifted traffic (drift -> exit 2)
drift-demo:
	rm -f logs/predictions.jsonl
	python -m churn.train
	python -m churn.simulate_traffic --n 2000 && python -m churn.drift || true
	python -m churn.simulate_traffic --n 2000 --drift && python -m churn.drift || echo "drift detected -> retraining would be triggered"
