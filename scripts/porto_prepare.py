"""Convert the ECML/PKDD 2015 Porto taxi CSV to DeepTTE JSONL shards.

Dataset (no auth needed):
https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip

Usage:
    uv run python scripts/porto_prepare.py --csv /path/to/train.csv \
        --out data/porto --max-trips 220000 --eval-trips 10000 --test-trips 10000

Porto trips are sampled every 15 SECONDS (time-sampled). Left as-is, that
LEAKS THE LABEL: total time == (n_points - 1) * 15, so a sequence model can
"predict" travel time by counting its input points (we measured 2% eval
loss before catching this). We therefore RESAMPLE every trajectory onto an
equal-DISTANCE grid (RESAMPLE_KM spacing) — point count then encodes trip
distance (a legitimate input), and per-point time gaps become the genuinely
unknown quantity. This is why the original DeepTTE README requires
distance-resampled GPS points.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deeptte.geo import haversine_km

SHARD_SIZE = 25_000
RESAMPLE_KM = 0.2   # spatial grid spacing, ~Chengdu's mean point gap
MIN_POINTS = 8      # after resampling (=> trips >= ~1.4 km)


def resample_by_distance(polyline, step_km):
    """Interpolate a 15s-sampled polyline onto an equal-distance grid.

    Returns (lngs, lats, time_gap, dist_gap) at grid points, or None if the
    trip is too short. time_gap is linearly interpolated cumulative seconds —
    the varying quantity the model must learn; dist_gap is the grid itself.
    """
    lngs = np.array([p[0] for p in polyline])
    lats = np.array([p[1] for p in polyline])
    cum_d = np.zeros(len(polyline))
    for i in range(1, len(polyline)):
        cum_d[i] = cum_d[i - 1] + haversine_km(lngs[i - 1], lats[i - 1], lngs[i], lats[i])
    cum_t = np.arange(len(polyline)) * 15.0

    grid = np.arange(0.0, cum_d[-1], step_km)
    if len(grid) < MIN_POINTS:
        return None
    grid = np.append(grid, cum_d[-1])  # keep the exact destination

    return (
        np.interp(grid, cum_d, lngs).tolist(),
        np.interp(grid, cum_d, lats).tolist(),
        np.interp(grid, cum_d, cum_t).tolist(),
        grid.tolist(),
    )


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

    resampled = resample_by_distance(polyline, RESAMPLE_KM)
    if resampled is None:
        return None
    lngs, lats, time_gap, dist_gap = resampled
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
        "lngs": lngs,
        "lats": lats,
        "states": [1.0] * len(lngs),
        "time_gap": time_gap,
        "dist_gap": dist_gap,
    }


def compute_stats(trips):
    """Match Chengdu semantics: dist_gap/time_gap stats are over per-point
    DELTAS (the stored sequences are cumulative); lngs/lats over all points;
    dist/time over trips. Stds are floored at 1% of |mean| — equal-distance
    resampling makes dist_gap deltas near-constant, and a raw std of ~0 would
    divide-by-zero in normalization / NaN the local-label math."""
    lngs = np.concatenate([t["lngs"] for t in trips])
    lats = np.concatenate([t["lats"] for t in trips])
    d_deltas = np.concatenate([np.diff(t["dist_gap"]) for t in trips])
    t_deltas = np.concatenate([np.diff(t["time_gap"]) for t in trips])
    dist = np.array([t["dist"] for t in trips])
    time = np.array([t["time"] for t in trips])

    def ms(x):
        mean, std = float(np.mean(x)), float(np.std(x))
        return [mean, max(std, 0.01 * abs(mean), 1e-6)]

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
    all_lngs = np.concatenate([t["lngs"] for t in train])
    all_lats = np.concatenate([t["lats"] for t in train])
    meta = {
        "stats": stats,
        # actual training bounding box — used by serving to refuse
        # out-of-coverage requests (σ-based bounds get wrecked by GPS glitches)
        "coverage": {
            "lng": [float(np.percentile(all_lngs, 0.01)), float(np.percentile(all_lngs, 99.99))],
            "lat": [float(np.percentile(all_lats, 0.01)), float(np.percentile(all_lats, 99.99))],
        },
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
