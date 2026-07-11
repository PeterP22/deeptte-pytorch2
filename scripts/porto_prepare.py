"""Convert the ECML/PKDD 2015 Porto taxi CSV to DeepTTE JSONL shards.

Dataset (no auth needed):
https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip

Usage:
    uv run python scripts/porto_prepare.py --csv /path/to/train.csv \
        --out data/porto --max-trips 220000 --eval-trips 10000 --test-trips 10000

Porto trips are sampled every 15 SECONDS (time-sampled), unlike Chengdu's
distance-resampled points. Consequence: every local window spans exactly
(kernel_size-1)*15 s, making the auxiliary local task near-trivial — use a
small --alpha when training on this dataset.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np

SHARD_SIZE = 25_000


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371


def convert_row(row, driver_ids):
    """Return a DeepTTE trip dict, or None if the row fails the filters."""
    if row["MISSING_DATA"] == "True":
        return None
    polyline = json.loads(row["POLYLINE"])
    n = len(polyline)
    if not 10 <= n <= 480:
        return None
    time_total = (n - 1) * 15.0
    if not 120 <= time_total <= 7200:
        return None

    dist_gap = [0.0]
    for (lng1, lat1), (lng2, lat2) in zip(polyline, polyline[1:]):
        dist_gap.append(dist_gap[-1] + haversine_km(lng1, lat1, lng2, lat2))
    if not 0.5 <= dist_gap[-1] <= 100:
        return None

    taxi = row["TAXI_ID"]
    if taxi not in driver_ids:
        driver_ids[taxi] = len(driver_ids)

    # UTC so the prepared dataset is identical on any machine
    start = datetime.fromtimestamp(int(row["TIMESTAMP"]), tz=timezone.utc)

    return {
        "driverID": driver_ids[taxi],
        "dateID": start.day - 1,
        "weekID": start.weekday(),
        "timeID": start.hour * 60 + start.minute,
        "dist": dist_gap[-1],
        "time": time_total,
        "lngs": [p[0] for p in polyline],
        "lats": [p[1] for p in polyline],
        "states": [1.0] * n,
        "time_gap": [15.0 * i for i in range(n)],
        "dist_gap": dist_gap,
    }


def compute_stats(trips):
    """Match Chengdu semantics: dist_gap/time_gap stats are over per-point
    DELTAS (the stored sequences are cumulative); lngs/lats over all points;
    dist/time over trips. Stds are floored at 1.0 — Porto's exact 15 s
    sampling makes the time_gap delta std 0.0, which would divide-by-zero in
    normalization and NaN the local-label math."""
    lngs = np.concatenate([t["lngs"] for t in trips])
    lats = np.concatenate([t["lats"] for t in trips])
    d_deltas = np.concatenate([np.diff(t["dist_gap"]) for t in trips])
    t_deltas = np.concatenate([np.diff(t["time_gap"]) for t in trips])
    dist = np.array([t["dist"] for t in trips])
    time = np.array([t["time"] for t in trips])

    def ms(x):
        return [float(np.mean(x)), float(max(np.std(x), 1.0))]

    return {
        "lngs": ms(lngs), "lats": ms(lats),
        "dist_gap": ms(d_deltas), "time_gap": ms(t_deltas),
        "dist": ms(dist), "time": ms(time),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="data/porto")
    parser.add_argument("--max-trips", type=int, default=220_000)
    parser.add_argument("--eval-trips", type=int, default=10_000)
    parser.add_argument("--test-trips", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    driver_ids = {}
    kept, seen = [], 0

    csv.field_size_limit(sys.maxsize)  # POLYLINE fields are long
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            seen += 1
            trip = convert_row(row, driver_ids)
            if trip is None:
                continue
            # reservoir sample to max_trips while streaming
            if len(kept) < args.max_trips:
                kept.append(trip)
            else:
                j = rng.integers(0, seen)
                if j < args.max_trips:
                    kept[j] = trip
            if seen % 200_000 == 0:
                print(f"  scanned {seen:,} rows, kept {len(kept):,}")

    print(f"scanned {seen:,} rows, kept {len(kept):,} trips, {len(driver_ids)} drivers")
    rng.shuffle(kept)

    n_eval, n_test = args.eval_trips, args.test_trips
    test, eval_, train = kept[:n_test], kept[n_test:n_test + n_eval], kept[n_test + n_eval:]
    if not train:
        raise SystemExit(f"no training trips left after split "
                         f"(kept {len(kept)}, eval {n_eval}, test {n_test})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, trips):
        with open(out / name, "w") as f:
            for t in trips:
                f.write(json.dumps(t) + "\n")

    train_files = []
    for i in range(0, len(train), SHARD_SIZE):
        name = f"train_{i // SHARD_SIZE:02d}"
        write(name, train[i:i + SHARD_SIZE])
        train_files.append(name)
    write("eval", eval_)
    write("test", test)

    stats = compute_stats(train)
    assert all(s[1] > 0 for s in stats.values()), "zero std slipped through"
    meta = {
        "stats": stats,
        "train_files": train_files,
        "eval_files": ["eval"],
        "test_files": ["test"],
        "n_train": len(train), "n_eval": len(eval_), "n_test": len(test),
        "n_drivers": len(driver_ids),
    }
    (out / "stats.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(train_files)} train shards + eval + test + stats.json to {out}/")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
