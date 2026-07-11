import numpy as np
import torch

from deeptte.config import Config
from deeptte.data import LengthBucketSampler, TripDataset, collate_fn, get_loader

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
