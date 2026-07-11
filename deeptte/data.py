"""Data pipeline: JSON-lines trips -> padded, normalized tensor batches.

Port of the original data_loader.py + utils.py padding logic.

DeepETA analogue: this is the feature-encoding stage. DeepTTE feeds raw
(normalized) GPS sequences; DeepETA instead bucketizes tabular features and
embeds them. Same job — turning a trip into model-ready numbers.
"""
import json
import math
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

STAT_KEYS = ("dist", "time")
INFO_KEYS = ("driverID", "dateID", "weekID", "timeID")
TRAJ_KEYS = ("lngs", "lats", "states", "time_gap", "dist_gap")

GEO_VOCAB = 16384
GEO_RESOLUTIONS = (("fine", 0.01), ("coarse", 0.05))  # degrees: ~1 km / ~5 km


def _cell(lng, lat, res):
    """Deterministic spatial-hash bucket for a grid cell.

    math.floor (not int()) so cells are stable across the sign boundary —
    Porto longitudes are negative. DeepETA analogue: multi-resolution geohash
    embeddings with independent hash functions.
    """
    x, y = math.floor(lng / res), math.floor(lat / res)
    return (x * 73856093 ^ y * 19349663) % GEO_VOCAB


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

    # origin/destination grid cells (raw coords, before normalization); the
    # model only embeds these when geohash is enabled
    for end, idx in (("o", 0), ("d", -1)):
        for suffix, res in GEO_RESOLUTIONS:
            attr[f"{end}_cell_{suffix}"] = torch.tensor(
                [_cell(item["lngs"][idx], item["lats"][idx], res) for item in batch],
                dtype=torch.long,
            )

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
        # partial (not a lambda) so num_workers>0 can pickle it on macOS
        collate_fn=partial(collate_fn, config=config),
        num_workers=num_workers,
    )
