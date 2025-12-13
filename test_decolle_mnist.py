#!/usr/bin/env python3
"""
Minimal DECOLLE-on-MNIST smoke test.

Runs a small training loop for the DECOLLE trainer on rate-coded MNIST.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = (repo_root / "src").as_posix()
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def train_one_epoch(trainer, train_loader, device):
    trainer.network.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, target in train_loader:
        data = data.transpose(0, 1).to(device)
        target = target.to(device)
        batch_size = data.size(1)

        trainer.reset()
        loss, pred = trainer.train_sample(data, target)

        total_loss += float(loss.item()) * batch_size
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
    }


def _sum_output_spikes(network, data_t):
    out = network(data_t)
    spk_list = out[0] if isinstance(out, (tuple, list)) else out
    return spk_list[-1]


def _accumulate_outputs(network, g_last, data_t):
    spk_out = _sum_output_spikes(network, data_t)  # [B, n_post]
    if g_last is None:
        return spk_out, None
    y_out = spk_out @ g_last.t()  # [B, n_classes]
    return spk_out, y_out


def evaluate(trainer, test_loader, device):
    import torch

    network = trainer.network
    g_last = getattr(trainer, "G", None)
    g_last = g_last[-1] if g_last is not None and len(g_last) > 0 else None

    network.eval()
    correct_spk = 0
    correct_y = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.transpose(0, 1).to(device)
            target = target.to(device)

            network.reset()
            spk_sum = None
            y_sum = None
            for t in range(data.size(0)):
                spk_out, y_out = _accumulate_outputs(network, g_last, data[t])
                spk_sum = spk_out if spk_sum is None else spk_sum + spk_out
                if y_out is not None:
                    y_sum = y_out if y_sum is None else y_sum + y_out

            preds_spk = spk_sum.argmax(dim=1)
            correct_spk += preds_spk.eq(target).sum().item()
            if y_sum is not None:
                preds_y = y_sum.argmax(dim=1)
                correct_y += preds_y.eq(target).sum().item()
            total += target.size(0)

    return {
        "spike_argmax": correct_spk / total if total > 0 else 0.0,
        "readout_argmax": correct_y / total if total > 0 else 0.0,
        "has_readout": y_sum is not None,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train DECOLLE on MNIST (smoke test).")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs to run.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--timesteps", type=int, default=25)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--device", type=str, default=None, help='e.g. "cpu", "cuda", "cuda:0"')
    p.add_argument(
        "--layers",
        type=str,
        default="784,256,10",
        help='Comma-separated layer sizes, e.g. "784,256,10".',
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    _add_src_to_path()

    import torch

    from datasets.get_loader import get_loader
    from networks.fc_network import FCNetwork
    from trainers.decolle_trainer import DECOLLETrainer

    args = parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")

    layer_sizes = [int(x) for x in args.layers.split(",") if x.strip()]
    if len(layer_sizes) < 2:
        raise SystemExit("--layers must have at least 2 sizes (in,out)")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.set_grad_enabled(False)

    train_loader, test_loader = get_loader("MNIST", args.batch_size, args.timesteps)
    network = FCNetwork(layer_sizes=layer_sizes, beta=args.beta)
    trainer = DECOLLETrainer(
        network=network,
        lr=args.lr,
        batch_size=args.batch_size,
        quant=False,
        use_optimizer=False,
        optimizer=None,
        surrogate="sigmoid",
        surrogate_scale=2.0,
    ).to(device)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(trainer, train_loader, device)
        test_metrics = evaluate(trainer, test_loader, device)
        dt = (time.perf_counter() - t0) * 1000.0

        print(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "test_acc_spike_argmax": test_metrics["spike_argmax"],
                "test_acc_readout_argmax": test_metrics["readout_argmax"],
                "epoch_ms": dt,
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
