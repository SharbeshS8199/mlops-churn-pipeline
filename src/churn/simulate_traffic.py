"""Send synthetic production traffic through the service (in-process, or over HTTP with --url).

    python -m churn.simulate_traffic --n 2000            # normal week
    python -m churn.simulate_traffic --n 2000 --drift    # price rise + customer-mix shift
"""

from __future__ import annotations

import argparse

from churn.data import CATEGORICAL, NUMERIC, synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--url", default=None, help="e.g. http://localhost:8000 to go through the API")
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()
    df = synthetic(args.n, seed=99, drift=args.drift)[NUMERIC + CATEGORICAL]
    records = df.to_dict(orient="records")
    if args.url:
        import requests

        for i in range(0, len(records), args.batch):
            requests.post(f"{args.url}/predict", json={"customers": records[i:i + args.batch]}, timeout=10).raise_for_status()
    else:
        from churn.serving import ChurnService

        svc = ChurnService()
        for i in range(0, len(records), args.batch):
            svc.predict(records[i:i + args.batch])
    print(f"sent {len(records)} {'drifted ' if args.drift else ''}requests")


if __name__ == "__main__":
    main()
