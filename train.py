"""Train the Fashion-MNIST CNN and report real metrics.

Usage:
    python train.py [--epochs 10] [--batch-size 128] [--lr 1e-3]

Outputs:
    runs/best_model.pt        best checkpoint (by test accuracy)
    runs/results.json         final metrics (test accuracy, per-class accuracy)
    docs/training_curves.png loss / accuracy curves
    docs/confusion_matrix.png confusion matrix of the best model on the test set
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_NAMES, FashionCNN

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
DOCS_DIR = ROOT / "docs"

# Standard Fashion-MNIST channel statistics.
MEAN, STD = (0.2860,), (0.3530,)


def get_loaders(batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train_set = datasets.FashionMNIST(
        DATA_DIR, train=True, download=True, transform=train_tf
    )
    test_set = datasets.FashionMNIST(
        DATA_DIR, train=False, download=True, transform=test_tf
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10
    confusion = [[0] * 10 for _ in range(10)]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.tolist(), pred.tolist()):
            confusion[t][p] += 1
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1
    per_class = {
        CLASS_NAMES[i]: round(per_class_correct[i] / per_class_total[i], 4)
        for i in range(10)
    }
    return correct / total, per_class, confusion


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], marker="o", label="Train")
    axes[1].plot(epochs, [a * 100 for a in history["test_acc"]], marker="s", label="Test")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "training_curves.png", dpi=120)
    plt.close(fig)


def plot_confusion(confusion):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="left", fontsize=8)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            v = confusion[i][j]
            if v > 0:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > 400 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    RUNS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    train_loader, test_loader = get_loaders(args.batch_size)
    model = FashionCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc, _, _ = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | "
            f"train acc {train_acc * 100:.2f}% | test acc {test_acc * 100:.2f}% | "
            f"elapsed {elapsed:.0f}s",
            flush=True,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), RUNS_DIR / "best_model.pt")

    total_time = time.time() - start

    # Final evaluation with the best checkpoint.
    model.load_state_dict(torch.load(RUNS_DIR / "best_model.pt", map_location=device))
    test_acc, per_class, confusion = evaluate(model, test_loader, device)
    plot_curves(history)
    plot_confusion(confusion)

    results = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "parameters": n_params,
        "training_seconds": round(total_time, 1),
        "best_test_accuracy": round(best_acc, 4),
        "final_test_accuracy_best_checkpoint": round(test_acc, 4),
        "per_class_accuracy": per_class,
        "history": history,
    }
    (RUNS_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Final results (best checkpoint) ===")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("Per-class accuracy:")
    for name, acc in per_class.items():
        print(f"  {name:<12} {acc * 100:.2f}%")
    print(f"Training time: {total_time:.0f}s for {args.epochs} epochs")
    print(f"Saved: {RUNS_DIR / 'best_model.pt'}, docs/training_curves.png, docs/confusion_matrix.png")


if __name__ == "__main__":
    main()
