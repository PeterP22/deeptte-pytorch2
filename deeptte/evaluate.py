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
