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
