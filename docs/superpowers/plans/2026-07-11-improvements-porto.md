# DeepTTE Improvements + Porto Scale-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut Chengdu test MAPE below 20% via training discipline + DeepETA-inspired flags, and scale to the Porto taxi dataset (200k trips) via a reproducible prepare script.

**Architecture:** All Tier-2 features are constructor flags on existing modules, defaulting to original behavior. Datasets become selectable via `Config.for_dataset`. Experiments run sequentially via a shell script, each into its own `checkpoints/<run>/` dir.

**Tech Stack:** unchanged (torch 2, numpy, tqdm, pytest). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-11-improvements-porto-design.md`

---

### Task 1: Tier 1 — training discipline in train.py

**Files:** Modify `deeptte/train.py`

- [ ] **Step 1:** Add args: `--patience` (int, default 8), `--run-name` (default: dataset name), `--dataset` (default "chengdu"), `--seed` (int, default 42), `--clip-norm` (float, default 1.0).
- [ ] **Step 2:** `torch.manual_seed(args.seed)` at start of main; `config = Config.for_dataset(args.dataset)` (Task 3 adds the method; until then keep `Config()` — implement Tasks 1–3 together before running).
- [ ] **Step 3:** Checkpoint dir becomes `checkpoints/<run-name>/`. Add scheduler + clipping + early stop:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
# in the batch loop, before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
# after eval each epoch:
scheduler.step(eval_loss)
# early stopping: track epochs_since_best; break when > args.patience, print why
```

- [ ] **Step 4:** Add the CLI args `--masked-attention` (store_true), `--dist-buckets` (int, default 0), `--geohash` (store_true) — but thread them into `DeepTTE(...)` only in Task 2 Step 6 (the constructor doesn't accept them yet; implement Tasks 1–3 before running the CLI).
- [ ] **Step 5:** `uv run pytest -q` still green. Commit `feat: tier-1 training discipline (early stop, plateau LR, clipping, runs)`.

### Task 2: Tier 2 — model flags (TDD)

**Files:** Modify `deeptte/models/attr.py`, `deeptte/models/spatio_temporal.py`, `deeptte/models/net.py`, `deeptte/data.py`; extend `tests/test_model.py`, `tests/test_data.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_data.py additions
def test_collate_emits_geo_cells():
    ds = TripDataset(DATA, CFG)
    attr, _ = collate_fn([ds[i] for i in range(4)], CFG)
    for key in ("o_cell_fine", "o_cell_coarse", "d_cell_fine", "d_cell_coarse"):
        assert attr[key].dtype == torch.long
        assert (attr[key] >= 0).all() and (attr[key] < 16384).all()

def test_cell_hash_deterministic_and_floors_negatives():
    from deeptte.data import _cell
    assert _cell(-8.61, 41.14, 0.01) == _cell(-8.61, 41.14, 0.01)
    assert _cell(-0.001, 0.001, 0.01) != _cell(0.001, 0.001, 0.01)  # floor, not truncate

# tests/test_model.py additions
def test_attr_dist_buckets_out_size():
    from deeptte.models.attr import Attr
    attr, _ = small_batch()
    net = Attr(dist_buckets=20)
    assert net.out_size() == 16 + 3 + 8 + 8  # embedding replaces raw scalar
    assert net(attr, CFG).shape == (4, net.out_size())

def test_attr_geohash_out_size():
    from deeptte.models.attr import Attr
    attr, _ = small_batch()
    net = Attr(geohash=True)
    assert net.out_size() == 28 + 32
    assert net(attr, CFG).shape == (4, net.out_size())

def test_masked_attention_shapes_and_normalization():
    from deeptte.models.attr import Attr
    from deeptte.models.spatio_temporal import SpatioTemporal
    attr, traj = small_batch()
    attr_net = Attr()
    st = SpatioTemporal(attr_size=attr_net.out_size(), masked_attention=True)
    _, _, pooled = st(traj, attr_net(attr, CFG), CFG)
    assert pooled.shape == (4, 128)
    assert torch.isfinite(pooled).all()

def test_net_flags_checkpoint_roundtrip(tmp_path):
    from deeptte.models.net import DeepTTE
    model = DeepTTE(masked_attention=True, dist_buckets=20, geohash=True)
    model.save_checkpoint(tmp_path / "c.pt")
    loaded = DeepTTE.from_checkpoint(tmp_path / "c.pt")
    assert loaded.hparams["dist_buckets"] == 20
```

- [ ] **Step 2:** Run — expect failures.
- [ ] **Step 3:** `data.py`: add deterministic spatial hash + emit cells in collate:

```python
import math
GEO_VOCAB = 16384

def _cell(lng, lat, res):
    """Deterministic spatial-hash bucket for a grid cell (math.floor: stable across sign)."""
    x, y = math.floor(lng / res), math.floor(lat / res)
    return (x * 73856093 ^ y * 19349663) % GEO_VOCAB

# in collate_fn, after INFO_KEYS loop (uses RAW coords, before normalization):
for end, idx in (("o", 0), ("d", -1)):
    for suffix, res in (("fine", 0.01), ("coarse", 0.05)):
        attr[f"{end}_cell_{suffix}"] = torch.tensor(
            [_cell(item["lngs"][idx], item["lats"][idx], res) for item in batch],
            dtype=torch.long,
        )
```

- [ ] **Step 4:** `attr.py`: constructor flags. `dist_buckets`: `register_buffer("dist_edges", torch.linspace(-2.5, 2.5, dist_buckets - 1))`, `nn.Embedding(dist_buckets, 8)`; forward does `torch.bucketize(attr["dist"], self.dist_edges)` → embed. **`attr["dist"]` arrives already normalized from the collate — do NOT normalize again** (re-normalizing would shift all trips into z ≈ [-2.9, -1.8], cramming the dataset into ~5 of 20 buckets). Enabling buckets thus also drops the double-normalization quirk. `geohash`: four `nn.Embedding(GEO_VOCAB, 8)` for the cell keys. `out_size()` becomes an instance method reflecting flags.
- [ ] **Step 5:** `spatio_temporal.py`: `masked_attention` flag; `attent_pooling(hiddens, attr_t, lens)`:

```python
if self.masked_attention:
    mask = (torch.arange(hiddens.size(1), device=hiddens.device)[None, :, None]
            < lens.to(hiddens.device)[:, None, None])
    alpha = alpha * mask
```

(`lens` = out_lens tensor from pad_packed_sequence; pass it in from forward. The mask multiply goes BETWEEN the `torch.exp` line and the sum-normalization — after normalization it does nothing.)
- [ ] **Step 6:** `net.py`: thread flags through `DeepTTE.__init__` into hparams + submodules. `entire_estimate` input size uses `self.attr_net.out_size()` (already instance-based after Step 4).
- [ ] **Step 7:** Full suite green. Commit `feat: tier-2 flags (masked attention, dist buckets, geohash embeddings)`.

### Task 3: Config.for_dataset + evaluate --dataset

**Files:** Modify `deeptte/config.py`, `deeptte/evaluate.py`; extend `tests/test_config.py`

- [ ] **Step 1: Failing test**

```python
def test_for_dataset_chengdu_is_default():
    from deeptte.config import Config
    assert Config.for_dataset("chengdu") == Config()

def test_for_dataset_unknown_raises():
    import pytest
    from deeptte.config import Config
    with pytest.raises(ValueError):
        Config.for_dataset("nyc")
```

- [ ] **Step 2:** Implement:

```python
@classmethod
def for_dataset(cls, name: str) -> "Config":
    if name == "chengdu":
        return cls()
    if name == "porto":
        import json as _json
        from pathlib import Path
        meta = _json.loads((Path("data/porto") / "stats.json").read_text())
        return cls(
            data_dir="data/porto",
            train_files=tuple(meta["train_files"]),
            eval_files=tuple(meta["eval_files"]),
            test_files=tuple(meta["test_files"]),
            stats={k: tuple(v) for k, v in meta["stats"].items()},
        )
    raise ValueError(f"unknown dataset: {name}")
```

(dataclass needs `eq=True` — default — for the equality test.)
- [ ] **Step 3:** `evaluate.py`: add `--dataset` arg → `Config.for_dataset`; results file defaults to `results/<dataset>-predictions.txt`.
- [ ] **Step 4:** Suite green. Commit `feat: dataset-selectable config (chengdu/porto)`.

### Task 4: Porto prepare script

**Files:** Create `scripts/porto_prepare.py`; modify `.gitignore` (add `data/porto/`, `data/porto.zip`, `*.csv`)

- [ ] **Step 1:** Implement script (stdlib + numpy only):

```python
"""Convert the ECML/PKDD 2015 Porto taxi CSV to DeepTTE JSONL shards.

Download (no auth): https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip
Usage: uv run python scripts/porto_prepare.py --csv train.csv --out data/porto \
           --max-trips 220000 --eval-trips 10000 --test-trips 10000 --seed 42
"""
# Core logic:
# - csv.reader over train.csv (POLYLINE parsed with json.loads)
# - filter per spec: MISSING_DATA=="False"; 10<=n<=480; 120<=(n-1)*15<=7200;
#   0.5 <= total_km <= 100
# - haversine(lng1,lat1,lng2,lat2) -> km (port of original utils.geo_distance)
# - fields per spec: driverID remapped contiguously, weekID/timeID/dateID from
#   datetime.fromtimestamp(TIMESTAMP), time=(n-1)*15.0, states=[1.0]*n,
#   time_gap=[15*i], dist_gap=cumulative haversine
# - reservoir-sample max_trips while streaming (seeded), then shuffle and split
#   train/eval/test; write train shards of 25k lines (train_00, train_01, ...),
#   plus files "eval" and "test"
# - stats computed on the TRAIN portion only: mean/std for dist_gap deltas?
#   -> match Chengdu semantics: stats for "dist_gap"/"time_gap" are over the
#   per-point gap VALUES (successive deltas), "lngs"/"lats" over all points,
#   "dist"/"time" over trips
# - write stats.json: {"stats": {...}, "train_files": [...], "eval_files": ["eval"],
#   "test_files": ["test"], "n_train":..., "n_eval":..., "n_test":...}
```

**Important semantic check before writing stats:** in Chengdu data, `dist_gap`/`time_gap` are CUMULATIVE sequences, but the normalization stats correspond to per-point deltas (time_gap_mean 43.9 s ≈ one sampling interval, not a cumulative mean). Porto must match: store cumulative sequences in the JSONL, compute `dist_gap`/`time_gap` stats over successive DELTAS.

**Zero-std guard (blocking bug otherwise):** Porto deltas for `time_gap` are exactly 15.0 for every point, so the delta std is 0.0 → division by zero in collate normalization AND `(k-1)*std("time_gap")` = 0 in the local-label math → NaN loss from the first batch. The prepare script must floor every stored std at 1.0 (`std = max(std, 1.0)`) with a comment explaining the Porto time-sampling cause. Sanity check in Step 3: assert no stored std is 0.

**Timezone:** derive `weekID`/`timeID`/`dateID` with `datetime.fromtimestamp(ts, tz=timezone.utc)` — machine-local time would make the prepared dataset irreproducible.
- [ ] **Step 2:** Download the zip (~500 MB) in the background; while it downloads, dry-run the script logic on a hand-made 5-line CSV fixture in the scratchpad to verify parsing/filtering/stats.
- [ ] **Step 3:** Run the full prepare; sanity-check output (`wc -l data/porto/*`, one JSON line eyeballed, stats.json plausible: time_gap delta mean == 15.0 exactly).
- [ ] **Step 4:** Smoke: `uv run python -m deeptte.train --dataset porto --epochs 1 --run-name porto-smoke` runs one epoch cleanly.
- [ ] **Step 5:** Commit `feat: Porto taxi dataset prepare script` (script + gitignore only, no data).

### Task 5: Experiment runner + Chengdu sweep

**Files:** Create `scripts/run_experiments.sh`

- [ ] **Step 1:**

```bash
#!/usr/bin/env bash
# Sequential experiment sweep; each run early-stops. Results appended as they land.
set -euo pipefail
run() {
  name=$1; shift
  echo "=== $name ==="
  uv run python -m deeptte.train --run-name "$name" --epochs 60 --patience 8 "$@"
  uv run python -m deeptte.evaluate --checkpoint "checkpoints/$name/best.pt" \
      --results-file "results/$name-predictions.txt" | tee -a results/experiments.log
}
mkdir -p results
run t1-a03
run t1-a01  --alpha 0.1
run t1-mask --masked-attention
run t1-buckets --dist-buckets 20
run t1-geo  --geohash
```

- [ ] **Step 2:** Launch in background; monitor `results/experiments.log` as runs land.
- [ ] **Step 3:** When all five finish: record a results table (README "Experiments" section), pick the winning flag combo.
- [ ] **Step 4:** Commit `feat: experiment sweep runner + chengdu sweep results`.

### Task 6: Porto training run

- [ ] **Step 1:** `uv run python -m deeptte.train --dataset porto --run-name porto-t1 --epochs 30 --patience 5 --alpha 0.1 <winning flags>` in background (~6–8 min/epoch at 200k trips; hours total — report incrementally).

  (alpha 0.1 regardless of Chengdu winner: Porto's constant 30 s local windows make the local task near-trivial — spec'd.)
- [ ] **Step 2:** Evaluate on Porto test split; compare against the same two baselines (evaluate.py computes them per dataset automatically).
- [ ] **Step 3:** README: Porto results row + a "Porto dataset" subsection (how to reproduce: download link + prepare command). Commit + push.
