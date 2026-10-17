"""CPU throughput benchmark. Uses only APIs present before and after the upgrade, so the
same script measures both: PYTHONPATH=<old checkout> uv run python scripts/benchmark.py ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from deeptte.config import Config
from deeptte.data import get_loader
from deeptte.models.net import DeepTTE


def _ctx(amp: str) -> contextlib.AbstractContextManager:
    return torch.autocast("cpu", dtype=torch.bfloat16) if amp == "bf16" else contextlib.nullcontext()


def train_throughput(batches, cfg, amp: str, fused: bool, compile_heads: bool,
                     warmup: int) -> float:
    torch.manual_seed(0)
    model = DeepTTE()
    if compile_heads:
        model.compile_heads()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused)
    model.train()
    trips, start = 0, 0.0
    for i, (attr, traj) in enumerate(batches):
        if i == warmup:
            start, trips = time.perf_counter(), 0
        with _ctx(amp):
            _, loss = model.eval_on_batch(attr, traj, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        trips += len(traj["lens"])
    return trips / (time.perf_counter() - start)


def infer_throughput(batches, cfg, amp: str, compile_heads: bool) -> float:
    torch.manual_seed(0)
    model = DeepTTE()
    if compile_heads:
        model.compile_heads()
    model.eval()
    with torch.inference_mode(), _ctx(amp):
        for attr, traj in batches[:3]:
            model.eval_on_batch(attr, traj, cfg)
        trips, start = 0, time.perf_counter()
        for attr, traj in batches:
            model.eval_on_batch(attr, traj, cfg)
            trips += len(traj["lens"])
    return trips / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-batches", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", default="fp32",
                        help="comma list from fp32,bf16,fused,compile")
    args = parser.parse_args()

    cfg = Config()
    train = [b for _, b in zip(range(args.train_batches),
                               get_loader(Path("data/train_00"), 64, cfg, seed=0))]
    test = list(get_loader(Path("data/test"), 64, cfg, seed=0))
    results: dict[str, dict[str, float]] = {}
    for mode in args.modes.split(","):
        amp = "bf16" if "bf16" in mode else "none"
        fused = "fused" in mode
        comp = "compile" in mode
        tr = [train_throughput(train, cfg, amp, fused, comp, args.warmup)
              for _ in range(args.repeats)]
        inf = [infer_throughput(test, cfg, amp, comp) for _ in range(args.repeats)]
        results[mode] = {"train_trips_per_s": round(statistics.median(tr), 1),
                         "infer_trips_per_s": round(statistics.median(inf), 1)}
        print(args.label, mode, results[mode], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "label": args.label, "torch": torch.__version__, "threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(), "repeats": args.repeats,
        "train_trips_measured": sum(len(t["lens"]) for _, t in train[args.warmup:]),
        "test_trips": sum(len(t["lens"]) for _, t in test), "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
