"""Train DeepTTE. Usage: uv run python -m deeptte.train [--epochs N] ..."""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import get_loader
from .models.net import DeepTTE

Batch = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | list[int]]]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(attr: dict, traj: dict, device: torch.device) -> tuple[dict, dict]:
    attr = {k: v.to(device, non_blocking=True) for k, v in attr.items()}
    traj = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in traj.items()}
    return attr, traj


def autocast(device: torch.device, amp: str) -> contextlib.AbstractContextManager:
    """bf16 autocast where requested; a no-op otherwise."""
    if amp == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_optimizer(model: torch.nn.Module, lr: float, device: torch.device) -> torch.optim.AdamW:
    fused = device.type in ("cpu", "cuda")
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=fused)


@dataclass
class EpochStats:
    loss: float
    trips: int
    seconds: float

    @property
    def trips_per_s(self) -> float:
        return self.trips / max(self.seconds, 1e-9)


def _batches(loaders: Iterable[DataLoader]) -> Iterator[Batch]:
    for loader in loaders:
        yield from loader


def train_one_epoch(model: DeepTTE, loaders: Iterable[DataLoader], optimizer: torch.optim.Optimizer,
                    config: Config, device: torch.device, amp: str = "none",
                    clip_norm: float = 1.0) -> EpochStats:
    model.train()
    total, batches, trips = 0.0, 0, 0
    start = time.perf_counter()
    for attr, traj in _batches(loaders):
        attr, traj = to_device(attr, traj, device)
        with autocast(device, amp):
            _, loss = model.eval_on_batch(attr, traj, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += loss.item()
        batches += 1
        trips += len(traj["lens"])
    return EpochStats(total / max(batches, 1), trips, time.perf_counter() - start)


@torch.inference_mode()
def evaluate_loss(model: DeepTTE, loaders: Iterable[DataLoader], config: Config,
                  device: torch.device, amp: str = "none") -> EpochStats:
    model.eval()
    total, batches, trips = 0.0, 0, 0
    start = time.perf_counter()
    for attr, traj in _batches(loaders):
        attr, traj = to_device(attr, traj, device)
        with autocast(device, amp):
            _, loss = model.eval_on_batch(attr, traj, config)
        total += loss.float().item()
        batches += 1
        trips += len(traj["lens"])
    return EpochStats(total / max(batches, 1), trips, time.perf_counter() - start)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", default="chengdu")
    parser.add_argument("--run-name", default=None,
                        help="checkpoints go to checkpoints/<run-name>/ (default: dataset)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8,
                        help="stop after this many epochs without eval improvement")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--masked-attention", action="store_true")
    parser.add_argument("--dist-buckets", type=int, default=0)
    parser.add_argument("--geohash", action="store_true")
    parser.add_argument("--amp", choices=["none", "bf16"], default="none",
                        help="bf16 autocast (slower than fp32 for this LSTM on CPU; see README)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the fixed-shape estimator heads")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--max-train-files", type=int, default=0, help="smoke runs only")
    parser.add_argument("--metrics-file", default=None,
                        help="default: <checkpoint-dir>/metrics.csv")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    config = Config.for_dataset(args.dataset)
    device = pick_device(args.device)
    run_name = args.run_name or args.dataset
    print(f"run {run_name}: training on {device}, torch {torch.__version__}, amp {args.amp}")

    model = DeepTTE(kernel_size=args.kernel_size, pooling_method=args.pooling,
                    alpha=args.alpha, masked_attention=args.masked_attention,
                    dist_buckets=args.dist_buckets, geohash=args.geohash).to(device)
    if args.compile:
        model.compile_heads()
    optimizer = make_optimizer(model, args.lr, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    train_files = config.train_files[: args.max_train_files or None]
    train_loaders = [get_loader(Path(config.data_dir) / name, args.batch_size, config,
                                seed=args.seed + i) for i, name in enumerate(train_files)]
    eval_loaders = [get_loader(Path(config.data_dir) / name, args.batch_size, config)
                    for name in config.eval_files]

    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else ckpt_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    best_eval = math.inf
    epochs_since_best = 0

    with open(metrics_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(["epoch", "train_loss", "eval_loss", "train_trips_per_s", "lr"])

        for epoch in range(1, args.epochs + 1):
            train = train_one_epoch(model, train_loaders, optimizer, config, device,
                                    args.amp, args.clip_norm)
            eval_loss = evaluate_loss(model, eval_loaders, config, device, args.amp).loss
            scheduler.step(eval_loss)
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch}: train {train.loss:.4f}  eval {eval_loss:.4f}  "
                  f"lr {lr:.2e}  {train.trips_per_s:.0f} trips/s")
            writer.writerow([epoch, f"{train.loss:.6f}", f"{eval_loss:.6f}",
                             f"{train.trips_per_s:.1f}", f"{lr:.2e}"])
            mf.flush()

            model.save_checkpoint(ckpt_dir / "last.pt")
            if eval_loss < best_eval:
                best_eval = eval_loss
                epochs_since_best = 0
                model.save_checkpoint(ckpt_dir / "best.pt")
                print(f"  new best (eval {eval_loss:.4f}) -> {ckpt_dir / 'best.pt'}")
            else:
                epochs_since_best += 1
                if epochs_since_best > args.patience:
                    print(f"early stop: no eval improvement in {args.patience} epochs "
                          f"(best {best_eval:.4f})")
                    break


if __name__ == "__main__":
    main()
