# DeepTTE Revival — Design Spec

**Date:** 2026-07-11
**Status:** Approved by Peter (2026-07-11)

## Purpose

Revive [UrbComp/DeepTTE](https://github.com/UrbComp/DeepTTE) — the code for the AAAI 2018 paper *"When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks"* — as a learning project to understand how deep-learning ETA models (like Uber's DeepETA) work. The original code is Python 2 + PyTorch ~0.3 and does not run on any modern setup.

Goals, in order:

1. **Port** the model to Python 3.12 + PyTorch 2.x, restructured as a modern package, with the architecture preserved exactly.
2. **Train** it locally (Apple Silicon Mac) on the Chengdu taxi trips bundled with the original repo: 5 train files × 3,600 trips (~18,000 total) plus a 1,400-trip test file.
3. **Evaluate** it with real metrics (MAPE / MAE / RMSE) on the bundled test set.
4. **Annotate** every module with docstrings that explain what it does and which Uber DeepETA concept it maps to — this is the learning layer.

**Explicitly out of scope for this spec:** deployment. Peter plans to deploy to Azure later (he has credits); that gets its own spec once training works. The only concession to it now: the trained model must load cleanly outside the training script (a `Net.from_checkpoint()`-style path), so the future serving layer doesn't need to import training code.

## Repo

- New **public GitHub repo** under Peter's account, working name `deeptte-pytorch2`.
- Fresh history (not a fork). README credits the original repo and paper, and states the mission: revive, modernize to current PyTorch, train, and benchmark; deployment to follow.
- The ~32 MB Chengdu sample data (5 train files + 1 test file, JSON-lines) ships in the repo, as the original did.

## Package layout

```
deeptte-pytorch2/
  deeptte/
    __init__.py
    models/
      __init__.py
      attr.py            # port of models/base/Attr.py
      geo_conv.py        # port of models/base/GeoConv.py
      spatio_temporal.py # port of models/base/SpatioTemporal.py
      net.py             # port of models/DeepTTE.py (top-level Net)
    data.py              # Dataset + collate fn; replaces data_loader.py + utils.py
    config.py            # typed dataclass config; replaces loose config.json
    train.py             # CLI entry point: uv run python -m deeptte.train
    evaluate.py          # CLI: metrics on test set, writes per-trip predictions
  data/                  # Chengdu trips copied from the original repo
  tests/
    test_data.py         # collate shapes, masks, normalization round-trip
    test_model.py        # one batch forward+backward, finite loss, no NaNs
  docs/superpowers/specs/
  pyproject.toml         # uv-managed; deps: torch>=2, numpy, tqdm, pytest
  README.md
```

## Model (architecture preserved 1:1)

- **Attr** — embeddings for `driverID`, `weekID`, `timeID` + normalized `dist`; concatenated attribute vector.
- **GeoConv** — per point, (lng, lat) is concatenated with a 2-dim embedding of the taxi `state`, mapped through `Linear(4, 16)` + tanh, then a 1D conv (kernel size 3) over the sequence; output concatenated with local `dist_gap`. Produces per-location features.
- **SpatioTemporal** — LSTM over Geo-Conv output with attribute vector injected; pooling over hidden states supports `attention` / `mean` / `last`, with `attention` as the default (matching the original constructor default and run example).
- **EntireEstimator / LocalEstimator** — residual FC head predicting total trip time; auxiliary head predicting per-segment local times. The local loss divides by `label + EPS` with `EPS = 10`; the entire loss has no epsilon — preserve both exactly.
- **Loss** — multi-task: `alpha * local_loss + (1 - alpha) * entire_loss`, both MAPE-style relative losses, with sequence masking for padded batches. Default `alpha = 0.3` (the original `Net.__init__` default; the original run example used 0.1 — both worth trying).

Port rules: `xrange` → `range`, print statements → logging/tqdm, `Variable`/`.data[0]` → plain tensors/`.item()`, `inspect.getargspec` → explicit kwargs, deprecated init calls → current `torch.nn.init`. Behavior-changing "improvements" to the architecture are out of scope.

## Data pipeline

- JSON-lines files, one trip per line, keys: `driverID`, `dateID`, `weekID`, `timeID`, `dist` (km), `time` (label, **seconds** — the original README says minutes but the data and `config.json` means are unambiguously seconds), `lngs`, `lats`, `states`, `time_gap`, `dist_gap`.
- **Split (adopting the original `config.json`):** train on `train_00`–`train_03`, hold out `train_04` as the eval set (drives `best.pt` selection), and reserve `test` for final metrics only.
- `torch.utils.data.Dataset` per file + custom `collate_fn` that sorts by length, pads sequences, and returns attr dict + traj dict + lengths mask (same semantics as the original `utils.pad_sequence`).
- Normalization constants (means/stds for dist, time, lngs, lats, gaps) come from `config.py`, seeded with the original `config.json` values.

## Training

- Device auto-detect: MPS → CUDA → CPU.
- AdamW, lr 1e-3 (original used Adam 1e-3; AdamW is the one deliberate modernization, noted in README).
- Checkpoint every epoch to `checkpoints/`, plus `best.pt` tracked by eval loss.
- tqdm progress; per-epoch train/eval loss appended to `metrics.csv` for plotting.
- Chosen defaults: batch 64 (the original `main.py` default; the run example used 10, which is needlessly slow on modern hardware), up to 100 epochs — but expect useful results in far fewer on ~14,400 training trips; early stopping is a manual decision, not automated (YAGNI).

## Evaluation

- `deeptte.evaluate --checkpoint <path>`: runs the test file, reports **MAPE, MAE, RMSE** — computed in seconds (the label unit) and reported in minutes for readability (explicit ÷60 at display time only) — and writes per-trip `(label, prediction)` pairs to a results file.
- Context for expectations, recorded in README: the paper reports ~11% MAPE trained on 5M+ trips; on ~14,400 training samples we expect materially worse. The gap is itself a documented learning point (data volume matters).

## Error handling

- Fail fast with clear messages on malformed data lines (report file + line number).
- Refuse to resume from a checkpoint whose config doesn't match the current model config.
- No elaborate recovery machinery — this is a training script, not a service.

## Testing

- `pytest` smoke tests, run before every commit:
  - `test_data.py`: collate produces expected shapes/masks; normalization is invertible.
  - `test_model.py`: one small batch forward + backward completes with finite loss on CPU.
- Full-training verification is empirical: loss decreases over epochs; evaluate produces sane MAPE.

## Milestones

1. Repo scaffold + data copied + README (revival mission, credits).
2. Data pipeline ported + tests green.
3. Model ported + smoke tests green.
4. Training run on Mac; loss curve recorded.
5. Evaluation metrics on test set; results in README.
6. (Later, separate spec) Azure deployment + serving API.
