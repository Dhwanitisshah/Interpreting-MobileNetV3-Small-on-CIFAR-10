"""Training and evaluation harness: SGD + cosine annealing, checkpointing, and
per-class/confusion-matrix evaluation artifacts consumed by every downstream
explanation script (Grad-CAM, faithfulness, robustness)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    limit_batches: Optional[int] = None,
) -> tuple:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def _run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    limit_batches: Optional[int] = None,
) -> tuple:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    device: torch.device,
    output_dir: str,
    limit_train_batches: Optional[int] = None,
    limit_val_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Train `model` for `cfg.train.epochs` epochs with SGD + cosine annealing.

    Writes `<output_dir>/checkpoints/{last,best}.pth` after every epoch (best
    by val_acc) and `<output_dir>/metrics.json` (full per-epoch history plus
    a summary) at the end. `limit_train_batches`/`limit_val_batches` truncate
    each epoch early, for smoke testing.
    """
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    output_path = Path(output_dir)
    checkpoints_dir = output_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(cfg.train.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = _run_train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, limit_train_batches
        )
        val_loss, val_acc = _run_val_epoch(
            model, val_loader, criterion, device, epoch, limit_val_batches
        )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "config": dict(cfg),
            "epoch": epoch,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(checkpoint, checkpoints_dir / "best.pth")

    summary = {"best_val_acc": best_val_acc, "best_epoch": best_epoch}
    metrics_out = {"history": history, "summary": summary}
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    return history


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Evaluate `model` on `loader`, returning overall/per-class accuracy, a
    full confusion matrix, and one record per example (index, true/pred
    label, confidence, correct) for downstream explanation scripts to select
    correctly/incorrectly classified images by index."""
    model.to(device)
    model.eval()

    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    records: List[Dict[str, Any]] = []
    index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            for i in range(labels.size(0)):
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())
                confidence = float(confidences[i].item())
                correct = true_label == pred_label

                confusion[true_label][pred_label] += 1
                records.append(
                    {
                        "index": index,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "correct": correct,
                    }
                )
                index += 1

    total = sum(sum(row) for row in confusion)
    correct_total = sum(confusion[i][i] for i in range(num_classes))
    overall_acc = correct_total / total if total > 0 else 0.0

    per_class_acc = []
    for i in range(num_classes):
        row_total = sum(confusion[i])
        per_class_acc.append(confusion[i][i] / row_total if row_total > 0 else 0.0)

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "confusion": confusion,
        "records": records,
    }


def save_eval_artifacts(eval_result: Dict[str, Any], output_dir: str) -> None:
    """Write `evaluate()`'s result to `<output_dir>/eval/`: predictions.json,
    confusion_matrix.json, and correct/incorrect_indices.json (the index
    lists Grad-CAM/sanity-check scripts use to pick visualization images)."""
    eval_dir = Path(output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "predictions.json", "w") as f:
        json.dump(eval_result["records"], f, indent=2)

    with open(eval_dir / "confusion_matrix.json", "w") as f:
        json.dump(
            {
                "confusion": eval_result["confusion"],
                "per_class_acc": eval_result["per_class_acc"],
                "overall_acc": eval_result["overall_acc"],
            },
            f,
            indent=2,
        )

    correct_indices = [r["index"] for r in eval_result["records"] if r["correct"]]
    incorrect_indices = [r["index"] for r in eval_result["records"] if not r["correct"]]
    total = len(eval_result["records"])
    assert len(correct_indices) + len(incorrect_indices) == total, (
        "correct/incorrect indices must partition the dataset"
    )

    with open(eval_dir / "correct_indices.json", "w") as f:
        json.dump(correct_indices, f, indent=2)

    with open(eval_dir / "incorrect_indices.json", "w") as f:
        json.dump(incorrect_indices, f, indent=2)
