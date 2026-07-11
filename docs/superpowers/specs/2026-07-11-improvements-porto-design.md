# DeepTTE Improvements + Porto Scale-Up — Design Spec

**Date:** 2026-07-11
**Status:** Approved by Peter (tiers approved in conversation; Porto explicitly requested)

## Purpose

Phase 1 landed at 24.55% test MAPE on Chengdu, overfitting ~14k trips (train 0.07 vs eval 0.23 by epoch 15). This phase closes the gap three ways, in order:

- **Tier 1 — training discipline:** early stopping (patience), ReduceLROnPlateau, gradient clipping, seeding, per-run checkpoint dirs; alpha sweep 0.3 vs 0.1.
- **Tier 2 — DeepETA-inspired model flags (each off by default, individually measurable):**
  - `--masked-attention`: mask padded positions in attention pooling (fixes the preserved quirk, as an experiment).
  - `--dist-buckets N`: bucketize the normalized total distance into N equal-z bins (linspace -2.5..2.5) and embed (8 dims) instead of the raw scalar. Simplification of DeepETA's quantile bucketization (documented as such; data-derived quantiles are a future refinement).
  - `--geohash`: embed origin/destination grid cells at two resolutions (0.01° ≈ 1 km, 0.05° ≈ 5 km), each hashed into a 16,384 vocab with a deterministic spatial hash, 8 dims each (+32 dims to Attr). Cells are computed in the collate (cheap, always emitted); the model uses them only when enabled.
- **Tier 3 — Porto taxi dataset:** the ECML/PKDD 2015 taxi trajectory dataset (~1.7M trips, UCI, no auth needed). A prepare script converts it to DeepTTE JSONL format, subsampled to 200k train / 10k eval / 10k test (configurable), with dataset-specific normalization stats. `Config.for_dataset("chengdu"|"porto")` selects data + stats; train/evaluate CLIs gain `--dataset`.

## Porto conversion rules

- Source columns: TAXI_ID, TIMESTAMP (trip start, unix), POLYLINE (list of [lng, lat] at 15 s intervals), MISSING_DATA.
- Filters: MISSING_DATA false; 10 ≤ points ≤ 480; travel time = (n_points − 1) × 15 s within [120 s, 7200 s]; total haversine distance within [0.5, 100] km.
- Derived fields: `driverID` = TAXI_ID remapped to contiguous ints; `weekID`/`timeID`/`dateID` from TIMESTAMP; `time_gap` = [0, 15, 30, …]; `dist_gap` = cumulative haversine (km); `states` = all 1.0 (Porto has no occupancy signal).
- **Known consequence, documented:** Porto is time-sampled (not distance-resampled as DeepTTE prefers), so local windows all span exactly 30 s and the auxiliary local loss becomes near-trivial — Porto runs should use small alpha. Recorded in README.
- Stats (means/stds for all normalized keys) computed on the train split only, written to `data/porto/stats.json` alongside shard file lists. `data/porto/` is gitignored (too big for the repo); the prepare script is the reproducibility story.

## Experiments (Chengdu, sequential, each early-stopped)

| Run | Flags |
|---|---|
| t1-a03 | Tier 1 defaults, alpha 0.3 |
| t1-a01 | alpha 0.1 |
| t1-mask | masked attention |
| t1-buckets | dist buckets (20) |
| t1-geo | geohash embeddings |

Each run: train with patience 8 (max 60 epochs), evaluate best.pt on the test set, append a row to a results table in the README. Then Porto trains with Tier 1 + whatever flags won.

## Out of scope

Deployment (next phase). Transformer encoder swap (possible later Tier 2b). Full 1.7M-trip training (subset first; scaling up is a rerun with a bigger `--max-trips`).

## Compatibility

- New model kwargs default to original behavior; Phase-1 checkpoints must still load via `from_checkpoint` (hparams dict gains keys only for new runs).
- Existing tests keep passing unchanged; new features get their own tests.
