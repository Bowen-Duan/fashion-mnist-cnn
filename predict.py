"""Load the trained checkpoint and classify a few test images.

Usage:
    python predict.py [--num 8] [--checkpoint runs/best_model.pt]

Prints the true vs predicted label for the first N test images and saves a
visual grid to docs/sample_predictions.png (green = correct, red = wrong).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from model import CLASS_NAMES, FashionCNN

ROOT = Path(__file__).resolve().parent
MEAN, STD = (0.2860,), (0.3530,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "best_model.pt")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint} — run train.py first.")

    device = torch.device("cpu")
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_set = datasets.FashionMNIST(ROOT / "data", train=False, download=True, transform=tf)

    n = min(args.num, len(test_set))
    images = torch.stack([test_set[i][0] for i in range(n)])
    labels = [test_set[i][1] for i in range(n)]
    with torch.no_grad():
        preds = model(images).argmax(dim=1).tolist()

    print(f"Predictions from {args.checkpoint.name} on the first {n} test images:")
    n_correct = 0
    for i, (true, pred) in enumerate(zip(labels, preds)):
        ok = true == pred
        n_correct += ok
        print(
            f"  image {i}: true = {CLASS_NAMES[true]:<12} "
            f"pred = {CLASS_NAMES[pred]:<12} {'OK' if ok else 'WRONG'}"
        )
    print(f"{n_correct}/{n} correct on this sample.")

    # Visual grid (denormalise for display).
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, img, true, pred in zip(axes, images, labels, preds):
        ax.imshow((img.squeeze(0) * STD[0] + MEAN[0]).clamp(0, 1), cmap="gray")
        ax.set_title(
            f"pred: {CLASS_NAMES[pred]}\ntrue: {CLASS_NAMES[true]}",
            fontsize=8,
            color="green" if true == pred else "red",
        )
        ax.axis("off")
    fig.tight_layout()
    out = ROOT / "docs" / "sample_predictions.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved grid: {out}")


if __name__ == "__main__":
    main()
