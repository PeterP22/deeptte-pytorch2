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


def run_eval(model, loaders, config, device):
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for loader in loaders:
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
    parser.add_argument("--metrics-file", default=None,
                        help="default: <checkpoint-dir>/metrics.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = Config.for_dataset(args.dataset)
    device = pick_device(args.device)
    run_name = args.run_name or args.dataset
    print(f"run {run_name}: training on {device}")

    model = DeepTTE(kernel_size=args.kernel_size, pooling_method=args.pooling,
                    alpha=args.alpha, masked_attention=args.masked_attention,
                    dist_buckets=args.dist_buckets, geohash=args.geohash).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # build loaders once — datasets stay parsed in memory across epochs, and
    # re-iterating a loader reshuffles via the sampler anyway
    train_loaders = [(name, get_loader(Path(config.data_dir) / name, args.batch_size, config))
                     for name in config.train_files]
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
            writer.writerow(["epoch", "train_loss", "eval_loss"])

        for epoch in range(1, args.epochs + 1):
            model.train()  # original bug fixed: was stuck in eval() after epoch 1
            total, batches = 0.0, 0
            for name, loader in train_loaders:
                for attr, traj in tqdm(loader, desc=f"epoch {epoch} {name}", leave=False):
                    attr, traj = to_device(attr, traj, device)
                    _, loss = model.eval_on_batch(attr, traj, config)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                    optimizer.step()
                    total += loss.item()
                    batches += 1
            train_loss = total / max(batches, 1)

            eval_loss = run_eval(model, eval_loaders, config, device)
            scheduler.step(eval_loss)
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch}: train {train_loss:.4f}  eval {eval_loss:.4f}  lr {lr:.2e}")
            writer.writerow([epoch, f"{train_loss:.6f}", f"{eval_loss:.6f}"])
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
