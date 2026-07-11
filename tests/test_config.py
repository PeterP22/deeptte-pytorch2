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
