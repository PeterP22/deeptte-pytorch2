# DeepTTE, revived for PyTorch 2

A modern revival of **[UrbComp/DeepTTE](https://github.com/UrbComp/DeepTTE)** — the official code for *"When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks"* (Wang, Zhang, Cao, Li & Zheng, **AAAI 2018**, [paper](https://ojs.aaai.org/index.php/AAAI/article/view/11877)).

All credit for the model design and the bundled Chengdu sample data goes to the original authors. The original code is Python 2 + PyTorch ~0.3 and no longer runs anywhere; this project ports it faithfully to **Python 3 + PyTorch 2**, trains it, and benchmarks it.

This is a learning project inspired by **Uber's DeepETA** ([blog](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/), [paper](https://arxiv.org/abs/2206.02127)) — every module's docstring maps it to its DeepETA analogue. DeepTTE is the 2018 research take (learn travel time end-to-end from the GPS trace); DeepETA is the production take (a linear transformer over tabular features that corrects a routing engine). Reading them side by side is the point.

## Architecture (unchanged from the paper)

```
trip metadata ──► Attr (driver/week/time embeddings + dist)
                        │
GPS sequence ──► GeoConv (state embed + Linear(4,16) + Conv1d) ──► 2-layer LSTM ──► attention pooling
                        │                                    │
                        └───────► LocalEstimator (per-window time, training-only)
                        ▼
                 EntireEstimator (residual MLP → total travel time)

loss = alpha * local + (1 - alpha) * entire      (both relative-error)
```

## What the port changed

**Fixed (deliberate deviations):**
- The original `main.py` never restored `model.train()` after per-epoch evaluation, so the paper's multi-task local loss silently applied only during epoch 1. Fixed.
- Unqualified `torch.squeeze()` crashed on batch size 1 — now explicit dims.
- `pack_padded_sequence(..., enforce_sorted=False)` for robustness (the length-bucketing sampler still sorts for padding efficiency).
- AdamW instead of Adam; `last.pt`/`best.pt` checkpoints instead of a timestamped file per epoch.
- The `time` label is in **seconds** — the original README said minutes, but the data and normalization constants are unambiguous.

**Preserved (faithful quirks):**
- `dist` is normalized twice (collate + Attr) — a fixed linear transform, kept for parity.
- Attention pooling has no padding mask (padded hiddens are zero; `exp(-0)=1` leaks a little weight into the denominator).
- Local loss uses `|pred-label| / (label + 10s)`; the entire-trip loss has no epsilon.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. Trains on Apple Silicon (MPS), CUDA, or CPU — auto-detected.

```bash
uv sync
uv run pytest                                     # smoke tests
uv run python -m deeptte.train --epochs 50        # ~40s/epoch on an M-series Mac
uv run python -m deeptte.evaluate --checkpoint checkpoints/best.pt
```

Training logs per-epoch train/eval loss to `checkpoints/metrics.csv`. Evaluation prints MAPE/MAE/RMSE against two sanity baselines (predict the mean time; constant-speed `time = dist / v̂`).

## Data

`data/` contains the original repo's Chengdu taxi sample: 5 train files x 3,600 trips + 1,400 test trips, JSON-lines. Split follows the original config: `train_00–03` train, `train_04` eval (drives `best.pt`), `test` for final metrics. Keys per trip: `driverID`, `dateID`, `weekID`, `timeID`, `dist` (km), `time` (label, seconds), and per-GPS-point sequences `lngs`, `lats`, `states`, `time_gap`, `dist_gap`.

The paper's ~11% MAPE used the full 5M+ trip dataset; with the ~18k-trip sample, expect materially worse — a hands-on lesson in why data volume is Uber's real moat.

## Results

*Training in progress — results will be recorded here.*

| Model | MAPE | MAE | RMSE |
|---|---|---|---|
| baseline: mean time | — | — | — |
| baseline: const speed | — | — | — |
| DeepTTE (this port) | — | — | — |

## Roadmap

- [x] Port to Python 3 / PyTorch 2
- [x] Train + evaluate on the Chengdu sample
- [ ] Deploy as an inference API (Azure)
- [ ] Results write-up

## Citation

```bibtex
@inproceedings{wang2018when,
  title={When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks},
  author={Wang, Dong and Zhang, Junbo and Cao, Wei and Li, Jian and Zheng, Yu},
  booktitle={AAAI},
  year={2018}
}
```
