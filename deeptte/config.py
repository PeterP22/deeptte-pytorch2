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
    coverage: dict | None = None  # {"lng": [lo, hi], "lat": [lo, hi]} training bbox

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

    @classmethod
    def for_dataset(cls, name: str) -> "Config":
        """Chengdu is built in; other datasets read stats.json written by
        their prepare script. Paths are relative to the repo root."""
        if name == "chengdu":
            return cls()
        if name == "porto":
            import json
            from pathlib import Path
            meta = json.loads((Path("data/porto") / "stats.json").read_text())
            return cls(
                data_dir="data/porto",
                train_files=tuple(meta["train_files"]),
                eval_files=tuple(meta["eval_files"]),
                test_files=tuple(meta["test_files"]),
                stats={k: tuple(v) for k, v in meta["stats"].items()},
                coverage=meta.get("coverage"),
            )
        raise ValueError(f"unknown dataset: {name}")
