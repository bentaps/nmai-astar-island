#!/usr/bin/env python3
"""
Train the residual neural process model.

Usage:
    python train_neural.py                    # default: validate on rounds 15-16
    python train_neural.py --epochs 300       # more training
    python train_neural.py --val-rounds 14 15 # custom validation
    python train_neural.py --device cpu       # force CPU
"""

import argparse
from astar.neural.train import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--episodes-per-round", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--val-rounds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="data/neural_model.pt")
    args = parser.parse_args()

    train_model(
        data_dir="data",
        model_save_path=args.save_path,
        n_epochs=args.epochs,
        episodes_per_round=args.episodes_per_round,
        lr=args.lr,
        val_rounds=args.val_rounds,
        device_name=args.device,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()
