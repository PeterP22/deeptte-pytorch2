# DeepTTE Revival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port DeepTTE (AAAI 2018) from Python 2 / PyTorch 0.3 to Python 3.12 / PyTorch 2.x, train it on the bundled ~18k Chengdu taxi trips on an Apple Silicon Mac, and evaluate it on the held-out test set.

**Success criteria (the "goal" for the trained model):** test-set MAPE beats both trivial baselines (predict-the-mean-time, and constant-speed `time = dist/v̂`) by a clear margin, and lands **under 20% MAPE**. (Paper: ~11% with 5M+ trips; we have ~14.4k training trips, so 15–20% is the realistic band.)

**Architecture:** Faithful 1:1 port of the model (Attr embeddings → GeoConv → 2-layer LSTM → attention pooling → residual FC head, multi-task loss `alpha*local + (1-alpha)*entire`), restructured as a modern `deeptte` package with typed config, proper Dataset/DataLoader, device auto-detect (MPS→CUDA→CPU), checkpointing, and CLI train/evaluate entry points.

**Tech Stack:** Python ≥3.11, uv, PyTorch ≥2.3, numpy, tqdm, pytest.

**Reference:** original code at `/Users/peterpreketes/uber/DeepTTE` (read-only). Spec: `docs/superpowers/specs/2026-07-11-deeptte-revival-design.md`.

## Porting decisions (locked in — do not relitigate during implementation)

1. **Preserve** the original's double-normalization of `dist` (normalized in collate AND again in Attr.forward). It's a quirk, but harmless (linear transform) and we want parity with the original. Comment it loudly.
2. **Preserve** the unmasked attention pooling (padded hiddens are 0; `exp(-0)=1` leaks a little weight to padding — original behavior, keep with a comment).
3. **Fix** (deliberate deviations, documented in README):
   - Original `main.py` never restored `model.train()` after per-epoch evaluation, so the multi-task local loss silently applied only in epoch 1. Our train loop calls `model.train()` at each epoch start.
   - `torch.squeeze()` without dim crashed on batch size 1 — use explicit dims (`.squeeze(1)`, `.squeeze(2)`).
   - `pack_padded_sequence(..., enforce_sorted=False)` instead of relying on the sampler's sort order for correctness (sampler still sorts for padding efficiency).
   - AdamW instead of Adam.
4. `time` labels are in **seconds**. Metrics computed in seconds, displayed also in minutes.
5. `dateID` is parsed into batches (parity with original) but unused by the model.
6. `states` is the only trajectory key NOT normalized (it feeds an Embedding).

## File structure

```
deeptte-pytorch2/
  pyproject.toml
  README.md
  .gitignore
  data/                      # copied from /Users/peterpreketes/uber/DeepTTE/data
  deeptte/
    __init__.py
    config.py                # normalization stats + file splits + helpers
    data.py                  # TripDataset, collate_fn, LengthBucketSampler, get_loader
    train.py                 # python -m deeptte.train
    evaluate.py              # python -m deeptte.evaluate (+ baselines)
    models/
      __init__.py
      attr.py                # Attr embeddings
      geo_conv.py            # GeoConv + get_local_seq
      spatio_temporal.py     # LSTM + pooling
      net.py                 # estimators + DeepTTE Net + checkpoint I/O
  tests/
    test_config.py
    test_data.py
    test_model.py
  checkpoints/               # gitignored
  results/                   # gitignored
```

---

### Task 1: Scaffold

**Files:** Create `pyproject.toml`, `.gitignore`, `README.md` (stub), `deeptte/__init__.py`, `deeptte/models/__init__.py`, copy `data/`.

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "deeptte"
version = "0.1.0"
description = "Modern PyTorch 2 revival of DeepTTE (AAAI 2018 travel-time estimation)"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "numpy>=1.26",
    "tqdm>=4.66",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["deeptte"]
```

- [ ] **Step 2: Create .gitignore** with: `__pycache__/`, `*.pyc`, `.venv/`, `checkpoints/`, `results/`, `.pytest_cache/`, `dist/`
- [ ] **Step 3: Copy data**: `cp -R /Users/peterpreketes/uber/DeepTTE/data ./data`
- [ ] **Step 4: Stub README.md** — title, one-paragraph mission (revival of UrbComp/DeepTTE, credit to Wang et al. AAAI 2018, link to original repo + paper + Uber DeepETA blog), "work in progress" note. Full README is Task 8.
- [ ] **Step 5: Empty `deeptte/__init__.py` and `deeptte/models/__init__.py`**
- [ ] **Step 6: `uv sync` — verify it resolves and installs torch.**
- [ ] **Step 7: Commit** `chore: scaffold deeptte package with Chengdu sample data`

### Task 2: Config

**Files:** Create `deeptte/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import torch
from deeptte.config import Config

def test_normalize_roundtrip():
    cfg = Config()
    x = torch.tensor([100.0, 2000.0])
    assert torch.allclose(cfg.unnormalize(cfg.normalize(x, "time"), "time"), x)

def test_splits_match_original():
    cfg = Config()
    assert cfg.train_files == ("train_00", "train_01", "train_02", "train_03")
    assert cfg.eval_files == ("train_04",)
    assert cfg.test_files == ("test",)

def test_time_stats_are_seconds_scale():
    cfg = Config()
    assert 1000 < cfg.mean("time") < 2000  # ~26 min in seconds
```

- [ ] **Step 2: Run `uv run pytest tests/test_config.py -v`** — expect FAIL (module missing).
- [ ] **Step 3: Implement `deeptte/config.py`**

```python
"""Normalization statistics and dataset splits for the Chengdu sample data.

Values are verbatim from the original repo's config.json. Note: `time` (the
label) and `time_gap` are in SECONDS — the original README claims minutes,
but the data says otherwise (mean trip time 1555.75 ~= 26 minutes).

DeepETA analogue: these are the hand-maintained feature statistics that
Uber replaced with quantile bucketization + learned embeddings.
"""
from dataclasses import dataclass, field

STATS: dict[str, tuple[float, float]] = {
    "dist_gap": (0.274716042312, 0.127051674693),
    "time_gap": (43.8756927994, 51.4811932987),
    "lngs": (104.05810954320589, 0.04988770679679998),
    "lats": (30.652312982784895, 0.04154695076189434),
    "dist": (9.578281194509781, 3.9656010701306283),
    "time": (1555.75269436, 646.373021152),
}

@dataclass(frozen=True)
class Config:
    data_dir: str = "data"
    train_files: tuple[str, ...] = ("train_00", "train_01", "train_02", "train_03")
    eval_files: tuple[str, ...] = ("train_04",)
    test_files: tuple[str, ...] = ("test",)
    stats: dict = field(default_factory=lambda: dict(STATS))

    def mean(self, key: str) -> float:
        return self.stats[key][0]

    def std(self, key: str) -> float:
        return self.stats[key][1]

    def normalize(self, x, key: str):
        mean, std = self.stats[key]
        return (x - mean) / std

    def unnormalize(self, x, key: str):
        mean, std = self.stats[key]
        return x * std + mean
```

- [ ] **Step 4: Run tests — expect PASS.**
- [ ] **Step 5: Commit** `feat: typed config with original normalization stats and splits`

### Task 3: Data pipeline

**Files:** Create `deeptte/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data.py
import numpy as np
import torch
from deeptte.config import Config
from deeptte.data import TripDataset, collate_fn, LengthBucketSampler, get_loader

CFG = Config()
DATA = "data/test"

def test_dataset_loads():
    ds = TripDataset(DATA, CFG)
    assert len(ds) == 1400
    trip = ds[0]
    assert set(["driverID", "weekID", "timeID", "dist", "time", "lngs", "lats",
                "states", "time_gap", "dist_gap"]).issubset(trip.keys())

def test_collate_shapes_and_padding():
    ds = TripDataset(DATA, CFG)
    batch = [ds[i] for i in range(4)]
    attr, traj = collate_fn(batch, CFG)
    max_len = max(len(item["lngs"]) for item in batch)
    assert traj["lngs"].shape == (4, max_len)
    assert attr["driverID"].dtype == torch.long
    assert attr["dist"].dtype == torch.float32
    assert traj["lens"] == [len(item["lngs"]) for item in batch]
    # states must NOT be normalized (feeds an Embedding): values in {0, 1} + padding
    assert set(traj["states"].unique().tolist()).issubset({0.0, 1.0})

def test_sampler_sorts_within_chunks():
    lengths = list(np.random.default_rng(0).integers(10, 100, size=500))
    sampler = LengthBucketSampler(lengths, batch_size=32, seed=1)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 500
    for b in batches:
        blens = [lengths[i] for i in b]
        assert blens == sorted(blens, reverse=True)

def test_loader_yields_batches():
    loader = get_loader(DATA, batch_size=8, config=CFG)
    attr, traj = next(iter(loader))
    assert attr["time"].shape[0] == 8
```

- [ ] **Step 2: Run `uv run pytest tests/test_data.py -v`** — expect FAIL.
- [ ] **Step 3: Implement `deeptte/data.py`**

```python
"""Data pipeline: JSON-lines trips -> padded, normalized tensor batches.

Port of the original data_loader.py + utils.py padding logic.

DeepETA analogue: this is the feature-encoding stage. DeepTTE feeds raw
(normalized) GPS sequences; DeepETA instead bucketizes tabular features and
embeds them. Same job — turning a trip into model-ready numbers.
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

STAT_KEYS = ("dist", "time")
INFO_KEYS = ("driverID", "dateID", "weekID", "timeID")
TRAJ_KEYS = ("lngs", "lats", "states", "time_gap", "dist_gap")


class TripDataset(Dataset):
    """One JSON-lines file of trips."""

    def __init__(self, path, config):
        self.config = config
        self.trips = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.trips.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: malformed trip JSON: {e}") from e
        self.lengths = [len(t["lngs"]) for t in self.trips]

    def __getitem__(self, idx):
        return self.trips[idx]

    def __len__(self):
        return len(self.trips)


def collate_fn(batch, config):
    """Build attr dict (per-trip scalars) and traj dict (padded sequences).

    Quirk preserved from the original: normalization is applied to the whole
    padded array, so padding positions hold (0 - mean)/std, not 0. Downstream
    packing (pack_padded_sequence) discards them.
    """
    lens = np.asarray([len(item["lngs"]) for item in batch])

    attr, traj = {}, {}
    for key in STAT_KEYS:
        x = torch.tensor([item[key] for item in batch], dtype=torch.float32)
        attr[key] = config.normalize(x, key)
    for key in INFO_KEYS:
        attr[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)

    mask = np.arange(lens.max()) < lens[:, None]
    for key in TRAJ_KEYS:
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate([np.asarray(item[key], dtype=np.float32) for item in batch])
        if key != "states":  # states feeds an Embedding; keep raw 0/1
            padded = config.normalize(padded, key)
        traj[key] = torch.from_numpy(padded)

    traj["lens"] = lens.tolist()
    return attr, traj


class LengthBucketSampler(Sampler):
    """Shuffle, then sort by descending length within chunks of batch_size*100.

    Port of the original BatchSampler: batches of similar-length trips
    minimize padding waste. Chunk size is a multiple of batch_size, so every
    batch comes from within one sorted chunk (descending order guaranteed).
    """

    def __init__(self, lengths, batch_size, seed=None):
        self.lengths = lengths
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = self.rng.permutation(len(self.lengths)).tolist()
        chunk = self.batch_size * 100
        for i in range(0, len(indices), chunk):
            indices[i:i + chunk] = sorted(
                indices[i:i + chunk], key=lambda j: self.lengths[j], reverse=True
            )
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def get_loader(path, batch_size, config, num_workers=0, seed=None):
    dataset = TripDataset(path, config)
    sampler = LengthBucketSampler(dataset.lengths, batch_size, seed=seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: collate_fn(b, config),
        num_workers=num_workers,
    )
```

- [ ] **Step 4: Run tests — expect PASS.**
- [ ] **Step 5: Commit** `feat: port data pipeline (dataset, collate, length-bucket sampler)`

### Task 4: Attr + GeoConv modules

**Files:** Create `deeptte/models/attr.py`, `deeptte/models/geo_conv.py`, `tests/test_model.py` (first half)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import torch
from deeptte.config import Config
from deeptte.data import TripDataset, collate_fn

CFG = Config()

def small_batch(n=4):
    ds = TripDataset("data/test", CFG)
    return collate_fn([ds[i] for i in range(n)], CFG)

def test_attr_output_size():
    from deeptte.models.attr import Attr
    attr, _ = small_batch()
    net = Attr()
    out = net(attr, CFG)
    assert out.shape == (4, net.out_size())
    assert net.out_size() == 16 + 3 + 8 + 1  # driver + week + time embeddings + dist

def test_attr_batch_of_one():
    from deeptte.models.attr import Attr
    attr, _ = small_batch(n=1)
    out = Attr()(attr, CFG)
    assert out.shape == (1, 28)  # original crashed here (unqualified squeeze)

def test_geo_conv_output_shape():
    from deeptte.models.geo_conv import GeoConv
    _, traj = small_batch()
    net = GeoConv(kernel_size=3, num_filter=32)
    out = net(traj, CFG)
    max_len = traj["lngs"].shape[1]
    assert out.shape == (4, max_len - 2, 33)  # T-k+1 windows, num_filter+1 features
```

- [ ] **Step 2: Run `uv run pytest tests/test_model.py -v`** — expect FAIL.
- [ ] **Step 3: Implement `deeptte/models/attr.py`**

```python
"""Attribute component: embeds per-trip metadata into a dense vector.

DeepETA analogue: DeepETA embeds ALL features this way (including
bucketized continuous ones) before feeding them to a linear transformer.
Here only driverID / weekID / timeID are embedded and total distance is
appended as a raw scalar.
"""
import torch
import torch.nn as nn

EMBED_DIMS = (("driverID", 24000, 16), ("weekID", 7, 3), ("timeID", 1440, 8))


class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        for name, dim_in, dim_out in EMBED_DIMS:
            self.add_module(name + "_em", nn.Embedding(dim_in, dim_out))

    @staticmethod
    def out_size() -> int:
        return sum(dim_out for _, _, dim_out in EMBED_DIMS) + 1  # +1 for dist

    def forward(self, attr, config):
        em_list = []
        for name, _, _ in EMBED_DIMS:
            embed = getattr(self, name + "_em")
            em_list.append(embed(attr[name].view(-1, 1)).squeeze(1))

        # Quirk preserved from the original: `dist` was already normalized in
        # collate_fn; the original normalizes it a second time here. Kept for
        # parity (it is just a fixed linear transform of the input).
        dist = config.normalize(attr["dist"], "dist")
        em_list.append(dist.view(-1, 1))

        return torch.cat(em_list, dim=1)
```

- [ ] **Step 4: Implement `deeptte/models/geo_conv.py`**

```python
"""Geo-Conv: learns local spatial features from the GPS point sequence.

Each point = (lng, lat, taxi-state embedding) -> Linear(4,16) -> tanh, then a
1D convolution over the sequence produces one feature vector per length-k
window, concatenated with the (normalized) distance covered by that window.

DeepETA analogue: DeepETA never sees raw GPS traces — it encodes origin /
destination as multi-resolution geohash embeddings. Geo-Conv is the
route-based alternative: convolve over the actual path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_local_seq(full_seq, kernel_size, mean, std):
    """Windowed difference: value covered by each length-k window of the trip.

    E.g. for dist_gap (cumulative), local[i] = full[i+k-1] - full[i].
    Output is re-normalized with the window-level mean/std.
    """
    local_seq = full_seq[:, kernel_size - 1:] - full_seq[:, :-(kernel_size - 1)]
    return (local_seq - mean) / std


class GeoConv(nn.Module):
    def __init__(self, kernel_size=3, num_filter=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, num_filter, kernel_size)

    def forward(self, traj, config):
        lngs = traj["lngs"].unsqueeze(2)
        lats = traj["lats"].unsqueeze(2)
        states = self.state_em(traj["states"].long())

        locs = torch.cat((lngs, lats, states), dim=2)
        locs = torch.tanh(self.process_coords(locs)).permute(0, 2, 1)
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # distance covered by each conv window, as an extra feature channel
        local_dist = get_local_seq(
            traj["dist_gap"], self.kernel_size,
            config.mean("dist_gap"), config.std("dist_gap"),
        )
        return torch.cat((conv_locs, local_dist.unsqueeze(2)), dim=2)
```

- [ ] **Step 5: Run tests — expect PASS.**

**Note on `test_attr_batch_of_one`:** `traj["states"].long()` after padding-normalization skip means states are exactly 0/1 — if this test fails on `.long()` values outside {0,1}, the collate normalization exclusion is broken, not this module.

- [ ] **Step 6: Commit** `feat: port Attr and GeoConv modules`

### Task 5: SpatioTemporal module

**Files:** Create `deeptte/models/spatio_temporal.py`, extend `tests/test_model.py`

- [ ] **Step 1: Write failing tests (append to tests/test_model.py)**

```python
def test_spatio_temporal_shapes():
    from deeptte.models.attr import Attr
    from deeptte.models.spatio_temporal import SpatioTemporal
    attr, traj = small_batch()
    attr_net = Attr()
    st = SpatioTemporal(attr_size=attr_net.out_size())
    attr_t = attr_net(attr, CFG)
    packed_hiddens, lens, pooled = st(traj, attr_t, CFG)
    assert pooled.shape == (4, 128)
    assert lens == [l - 2 for l in traj["lens"]]  # kernel_size 3 shrinks by 2

def test_spatio_temporal_mean_pooling():
    from deeptte.models.attr import Attr
    from deeptte.models.spatio_temporal import SpatioTemporal
    attr, traj = small_batch()
    attr_net = Attr()
    st = SpatioTemporal(attr_size=attr_net.out_size(), pooling_method="mean")
    _, _, pooled = st(traj, attr_net(attr, CFG), CFG)
    assert pooled.shape == (4, 128)
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `deeptte/models/spatio_temporal.py`**

```python
"""Spatio-temporal component: LSTM over Geo-Conv features + pooling.

The trip's per-window features (with the attribute vector appended to every
step) run through a 2-layer LSTM; hidden states are pooled to one vector.

Pooling options: 'attention' (default, matches original), 'mean'.
Attention here is over TIME STEPS of one trip. DeepETA's attention is over
FEATURES of one (tabular) trip — same mechanism, different axis, and that
difference is the heart of DeepTTE-vs-DeepETA.
"""
import torch
import torch.nn as nn

from .geo_conv import GeoConv


class SpatioTemporal(nn.Module):
    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method="attention"):
        super().__init__()
        if pooling_method not in ("attention", "mean"):
            raise ValueError(f"unsupported pooling_method: {pooling_method}")
        self.kernel_size = kernel_size
        self.pooling_method = pooling_method
        self.geo_conv = GeoConv(kernel_size=kernel_size, num_filter=num_filter)
        self.rnn = nn.LSTM(
            input_size=num_filter + 1 + attr_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        if pooling_method == "attention":
            self.attr2atten = nn.Linear(attr_size, 128)

    @staticmethod
    def out_size() -> int:
        return 128

    def mean_pooling(self, hiddens, lens):
        # padded hiddens are 0, so sum/len == masked mean
        summed = torch.sum(hiddens, dim=1)
        lens = lens.to(summed).unsqueeze(1)
        return summed / lens

    def attent_pooling(self, hiddens, attr_t):
        # attr_t arrives already unsqueezed to (B, 1, attr_size)
        attent = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)  # B x 128 x 1
        alpha = torch.exp(-torch.bmm(hiddens, attent))  # B x T x 1
        # Quirk preserved from the original: no explicit padding mask. Padded
        # hiddens are 0 so exp(-0)=1 leaks some weight into the denominator.
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return torch.bmm(hiddens.permute(0, 2, 1), alpha).squeeze(2)

    def forward(self, traj, attr_t, config):
        conv_locs = self.geo_conv(traj, config)

        # append the attribute vector to every time step
        attr_t = attr_t.unsqueeze(1)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size(-1),))
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)

        # conv with kernel k shrinks each trip by k-1 windows
        lens = [l - self.kernel_size + 1 for l in traj["lens"]]

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            conv_locs, lens, batch_first=True, enforce_sorted=False
        )
        packed_hiddens, _ = self.rnn(packed_inputs)
        hiddens, out_lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        if self.pooling_method == "mean":
            return packed_hiddens, lens, self.mean_pooling(hiddens, out_lens)
        return packed_hiddens, lens, self.attent_pooling(hiddens, attr_t)
```

- [ ] **Step 4: Run tests — expect PASS.**
- [ ] **Step 5: Commit** `feat: port SpatioTemporal (LSTM + attention/mean pooling)`

### Task 6: Estimators + Net + checkpoint I/O

**Files:** Create `deeptte/models/net.py`, extend `tests/test_model.py`

- [ ] **Step 1: Write failing tests (append)**

```python
def test_net_forward_backward_finite():
    from deeptte.models.net import DeepTTE
    attr, traj = small_batch(n=8)
    model = DeepTTE()
    model.train()
    pred_dict, loss = model.eval_on_batch(attr, traj, CFG)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

def test_net_eval_mode_returns_predictions():
    from deeptte.models.net import DeepTTE
    attr, traj = small_batch(n=8)
    model = DeepTTE()
    model.eval()
    with torch.no_grad():
        pred_dict, loss = model.eval_on_batch(attr, traj, CFG)
    assert pred_dict["pred"].shape == (8, 1)
    assert pred_dict["label"].shape == (8, 1)
    assert (pred_dict["label"] > 0).all()  # unnormalized seconds

def test_checkpoint_roundtrip(tmp_path):
    from deeptte.models.net import DeepTTE
    model = DeepTTE(kernel_size=3, num_filter=32, alpha=0.3)
    path = tmp_path / "ckpt.pt"
    model.save_checkpoint(path)
    loaded = DeepTTE.from_checkpoint(path)
    for a, b in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(a, b)
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `deeptte/models/net.py`**

```python
"""DeepTTE top-level network: multi-task travel-time estimation.

EntireEstimator: residual MLP head predicting TOTAL trip time from
(attribute vector, pooled spatio-temporal vector).
LocalEstimator: small MLP predicting the time of each local window from the
per-step LSTM hidden states (training-only auxiliary task).
Loss: alpha * local + (1 - alpha) * entire, both relative-error (MAPE-style).

DeepETA analogues: EntireEstimator ~ DeepETA's fully-connected decoder with
bias-adjustment layers; the relative-error loss plays the role DeepETA gives
its asymmetric Huber loss (robustness to outlier trips).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attr import Attr
from .geo_conv import get_local_seq
from .spatio_temporal import SpatioTemporal

EPS = 10  # seconds; keeps the local relative loss from exploding on tiny windows


class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super().__init__()
        self.input2hid = nn.Linear(input_size, hidden_size)
        self.residuals = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_final_fcs)
        )
        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)
        hidden = F.leaky_relu(self.input2hid(inputs))
        for layer in self.residuals:
            hidden = hidden + F.leaky_relu(layer(hidden))
        return self.hid2out(hidden)

    def eval_on_batch(self, pred, label, mean, std):
        label = label.view(-1, 1) * std + mean
        pred = pred * std + mean
        loss = torch.abs(pred - label) / label
        return {"label": label, "pred": pred}, loss.mean()


class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))
        hidden = F.leaky_relu(self.hid2hid(hidden))
        return self.hid2out(hidden)

    def eval_on_batch(self, pred, lens, label, mean, std):
        # pack the padded label sequence the same way the hidden states were
        # packed, so pred and label rows line up
        label = nn.utils.rnn.pack_padded_sequence(
            label, lens, batch_first=True, enforce_sorted=False
        ).data
        label = label.view(-1, 1) * std + mean
        pred = pred * std + mean
        # EPS in the denominator only (original behavior — preserve exactly)
        loss = torch.abs(pred - label) / (label + EPS)
        return loss.mean()


class DeepTTE(nn.Module):
    def __init__(self, kernel_size=3, num_filter=32, pooling_method="attention",
                 num_final_fcs=3, final_fc_size=128, alpha=0.3):
        super().__init__()
        self.hparams = dict(
            kernel_size=kernel_size, num_filter=num_filter,
            pooling_method=pooling_method, num_final_fcs=num_final_fcs,
            final_fc_size=final_fc_size, alpha=alpha,
        )
        self.kernel_size = kernel_size
        self.alpha = alpha

        self.attr_net = Attr()
        self.spatio_temporal = SpatioTemporal(
            attr_size=self.attr_net.out_size(),
            kernel_size=kernel_size,
            num_filter=num_filter,
            pooling_method=pooling_method,
        )
        self.entire_estimate = EntireEstimator(
            input_size=SpatioTemporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=num_final_fcs,
            hidden_size=final_fc_size,
        )
        self.local_estimate = LocalEstimator(input_size=SpatioTemporal.out_size())

        self._init_weight()

    def _init_weight(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr, config)
        # sptm_s: PackedSequence of hidden states; sptm_l: window counts;
        # sptm_t: pooled trip vector
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t, config)
        entire_out = self.entire_estimate(attr_t, sptm_t)
        if self.training:
            local_out = self.local_estimate(sptm_s.data)
            return entire_out, (local_out, sptm_l)
        return entire_out

    def eval_on_batch(self, attr, traj, config):
        if self.training:
            entire_out, (local_out, local_length) = self(attr, traj, config)
        else:
            entire_out = self(attr, traj, config)

        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(
            entire_out, attr["time"], config.mean("time"), config.std("time")
        )

        if not self.training:
            return pred_dict, entire_loss

        # local windows span (kernel_size - 1) gaps
        mean = (self.kernel_size - 1) * config.mean("time_gap")
        std = (self.kernel_size - 1) * config.std("time_gap")
        local_label = get_local_seq(traj["time_gap"], self.kernel_size, mean, std)
        local_loss = self.local_estimate.eval_on_batch(
            local_out, local_length, local_label, mean, std
        )
        return pred_dict, (1 - self.alpha) * entire_loss + self.alpha * local_loss

    def save_checkpoint(self, path):
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model
```

- [ ] **Step 4: Run `uv run pytest -v`** — ALL tests expect PASS.
- [ ] **Step 5: Commit** `feat: port estimators and DeepTTE net with checkpoint I/O`

### Task 7: Training CLI

**Files:** Create `deeptte/train.py`

- [ ] **Step 1: Implement `deeptte/train.py`** (verification is a real smoke run, not a unit test)

```python
"""Train DeepTTE. Usage: uv run python -m deeptte.train [--epochs N] ..."""
import argparse
import csv
import math
from pathlib import Path

import torch
from tqdm import tqdm

from .config import Config
from .data import get_loader
from .models.net import DeepTTE


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(attr, traj, device):
    attr = {k: v.to(device) for k, v in attr.items()}
    traj = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in traj.items()}
    return attr, traj


def run_eval(model, files, config, device, batch_size):
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for name in files:
            loader = get_loader(Path(config.data_dir) / name, batch_size, config)
            for attr, traj in loader:
                attr, traj = to_device(attr, traj, device)
                _, loss = model.eval_on_batch(attr, traj, config)
                total += loss.item()
                batches += 1
    return total / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics-file", default="checkpoints/metrics.csv")
    args = parser.parse_args()

    config = Config()
    device = pick_device(args.device)
    print(f"training on {device}")

    model = DeepTTE(kernel_size=args.kernel_size, pooling_method=args.pooling,
                    alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    metrics_path = Path(args.metrics_file)
    new_file = not metrics_path.exists()
    best_eval = math.inf

    with open(metrics_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(["epoch", "train_loss", "eval_loss"])

        for epoch in range(1, args.epochs + 1):
            model.train()  # original bug fixed: was stuck in eval() after epoch 1
            total, batches = 0.0, 0
            for name in config.train_files:
                loader = get_loader(Path(config.data_dir) / name, args.batch_size, config)
                for attr, traj in tqdm(loader, desc=f"epoch {epoch} {name}", leave=False):
                    attr, traj = to_device(attr, traj, device)
                    _, loss = model.eval_on_batch(attr, traj, config)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
                    batches += 1
            train_loss = total / max(batches, 1)

            eval_loss = run_eval(model, config.eval_files, config, device, args.batch_size)
            print(f"epoch {epoch}: train {train_loss:.4f}  eval {eval_loss:.4f}")
            writer.writerow([epoch, f"{train_loss:.6f}", f"{eval_loss:.6f}"])
            mf.flush()

            model.save_checkpoint(ckpt_dir / "last.pt")
            if eval_loss < best_eval:
                best_eval = eval_loss
                model.save_checkpoint(ckpt_dir / "best.pt")
                print(f"  new best (eval {eval_loss:.4f}) -> checkpoints/best.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run:** `uv run python -m deeptte.train --epochs 1 --batch-size 64`
  Expected: runs to completion on MPS (or CPU fallback), prints train/eval loss, writes `checkpoints/last.pt`, `checkpoints/best.pt`, `checkpoints/metrics.csv`. Train loss should be well below the untrained ~1.0 relative error by end of epoch 1.
  **If MPS errors on LSTM/packed sequences:** rerun with `--device cpu`, note it, continue (CPU is fine at this scale).
- [ ] **Step 3: Run full test suite** `uv run pytest -v` — expect PASS.
- [ ] **Step 4: Commit** `feat: training CLI with device auto-detect, checkpoints, metrics log`

### Task 8: Evaluation CLI + baselines

**Files:** Create `deeptte/evaluate.py`

- [ ] **Step 1: Implement `deeptte/evaluate.py`**

```python
"""Evaluate a checkpoint on the test set; compare against trivial baselines.

Usage: uv run python -m deeptte.evaluate --checkpoint checkpoints/best.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import get_loader
from .models.net import DeepTTE
from .train import pick_device, to_device


def metrics(pred, label):
    pred, label = np.asarray(pred), np.asarray(label)
    mape = float(np.mean(np.abs(pred - label) / label))
    mae = float(np.mean(np.abs(pred - label)))
    rmse = float(np.sqrt(np.mean((pred - label) ** 2)))
    return {"mape": mape, "mae_s": mae, "rmse_s": rmse}


def report(name, m):
    print(f"{name:>22}: MAPE {m['mape'] * 100:6.2f}%   "
          f"MAE {m['mae_s']:7.1f}s ({m['mae_s'] / 60:.2f} min)   "
          f"RMSE {m['rmse_s']:7.1f}s ({m['rmse_s'] / 60:.2f} min)")


def load_trips(config, files):
    trips = []
    for name in files:
        with open(Path(config.data_dir) / name) as f:
            trips += [json.loads(line) for line in f if line.strip()]
    return trips


def baselines(config):
    """Two sanity baselines the model must beat."""
    train = load_trips(config, config.train_files)
    test = load_trips(config, config.test_files)
    label = np.array([t["time"] for t in test])

    mean_time = np.mean([t["time"] for t in train])
    mean_pred = np.full_like(label, mean_time)

    # constant-speed: aggregate v = total dist / total time over training trips
    v = sum(t["dist"] for t in train) / sum(t["time"] for t in train)
    speed_pred = np.array([t["dist"] / v for t in test])

    report("baseline: mean time", metrics(mean_pred, label))
    report("baseline: const speed", metrics(speed_pred, label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-file", default="results/predictions.txt")
    args = parser.parse_args()

    config = Config()
    device = pick_device(args.device)
    model = DeepTTE.from_checkpoint(args.checkpoint).to(device)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for name in config.test_files:
            loader = get_loader(Path(config.data_dir) / name, args.batch_size, config)
            for attr, traj in loader:
                attr, traj = to_device(attr, traj, device)
                pred_dict, _ = model.eval_on_batch(attr, traj, config)
                preds += pred_dict["pred"].cpu().squeeze(1).tolist()
                labels += pred_dict["label"].cpu().squeeze(1).tolist()

    Path(args.results_file).parent.mkdir(exist_ok=True)
    with open(args.results_file, "w") as f:
        for l, p in zip(labels, preds):
            f.write(f"{l:.1f} {p:.1f}\n")

    baselines(config)
    report("DeepTTE", metrics(preds, labels))
    print(f"\nper-trip predictions -> {args.results_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify:** `uv run python -m deeptte.evaluate --checkpoint checkpoints/best.pt` (using the Task 7 smoke checkpoint). Expected: prints two baseline rows + DeepTTE row; after only 1 epoch DeepTTE may not beat baselines yet — that's fine, this step verifies plumbing, not accuracy.
- [ ] **Step 3: Commit** `feat: evaluation CLI with MAPE/MAE/RMSE and trivial baselines`

### Task 9: README + publish to GitHub

**Files:** Modify `README.md`

- [ ] **Step 1: Write the full README** — sections: What this is (revival of UrbComp/DeepTTE, full credit to Wang, Fu, Ye, Chen & Xie, AAAI 2018 paper link, original repo link); Why (learning project inspired by Uber's DeepETA blog + paper, links); What changed in the port (Py3/PyTorch 2, the three deliberate fixes, preserved quirks, seconds-not-minutes correction); Quickstart (`uv sync`, train command, evaluate command); Data format (from original README, with the seconds correction); Results table (placeholder, filled in Task 10); Roadmap (deployment later).
- [ ] **Step 2: Create public GitHub repo and push** (user pre-approved a public repo):

```bash
gh repo create deeptte-pytorch2 --public --source . --description "DeepTTE (AAAI 2018) travel-time estimation, revived for PyTorch 2 — learning project inspired by Uber's DeepETA" --push
```

- [ ] **Step 3: Commit + push** `docs: full README with credits, port notes, usage`

### Task 10: Full training run + results

- [ ] **Step 1: Train:** `uv run python -m deeptte.train --epochs 50` (background; monitor `checkpoints/metrics.csv`). Watch first epochs — if eval loss plateaus early, it's fine to stop manually.
- [ ] **Step 2: Evaluate best checkpoint:** `uv run python -m deeptte.evaluate --checkpoint checkpoints/best.pt`
- [ ] **Step 3: Check the goal:** DeepTTE MAPE < 20% and beats both baselines. Record actual numbers.
- [ ] **Step 4: Fill the README results table** with baseline + model metrics and epochs trained; note train time and device.
- [ ] **Step 5: Commit + push** `docs: training results on Chengdu sample data`
