"""Fire real test trips at a live DeepTTE endpoint and score the responses.

The labels are stripped before sending — the server predicts blind, and we
compare its answers against ground truth. A live replay of offline evaluation.

Usage:
    uv run python scripts/hit_endpoint.py --url https://<app>.azurecontainerapps.io \
        --dataset porto --n 50
"""
import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from deeptte.config import Config


def departure_iso(trip):
    """Reconstruct a plausible departure datetime matching the trip's
    weekID/timeID (the model only sees those two, so any matching date works)."""
    base = datetime(2026, 7, 6, tzinfo=timezone.utc)  # a Monday
    dep = base + timedelta(days=trip["weekID"],
                           minutes=trip["timeID"])
    return dep.isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config.for_dataset(args.dataset)
    test_file = Path(config.data_dir) / config.test_files[0]
    trips = [json.loads(l) for l in open(test_file) if l.strip()]
    random.Random(args.seed).shuffle(trips)
    trips = trips[:args.n]

    client = httpx.Client(timeout=30)
    health = client.get(f"{args.url}/healthz").json()
    print(f"endpoint healthy: checkpoint={health['checkpoint']} dataset={health['dataset']}\n")

    preds, labels = [], []
    for i, trip in enumerate(trips):
        payload = {
            "points": [[lng, lat] for lng, lat in zip(trip["lngs"], trip["lats"])],
            "departure_time": departure_iso(trip),
            "driver_id": trip["driverID"],
        }
        r = client.post(f"{args.url}/predict", json=payload)
        r.raise_for_status()
        eta = r.json()["eta_seconds"]
        preds.append(eta)
        labels.append(trip["time"])
        print(f"trip {i:3d}: actual {trip['time']:7.0f}s  predicted {eta:7.0f}s  "
              f"error {abs(eta - trip['time']) / trip['time'] * 100:5.1f}%")

    preds, labels = np.array(preds), np.array(labels)
    mape = np.mean(np.abs(preds - labels) / labels) * 100
    mae = np.mean(np.abs(preds - labels))
    print(f"\nlive endpoint over {len(trips)} trips: "
          f"MAPE {mape:.2f}%  MAE {mae:.0f}s ({mae / 60:.2f} min)")


if __name__ == "__main__":
    main()
