"""Recompute K=1..3 CelebA precision/recall metrics from cached checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from datasets.celeba.celeba_attacked import get_celeba_attacked_dataset
from models.vgg import get_vgg16


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/media/erogullari/datasets")
CLEAN_CHECKPOINT = Path("/media/erogullari/checkpoints/checkpoint_vgg16_celeba.pth")
CHECKPOINTS = sorted(
    (REPOSITORY_ROOT / "checkpoints" / "celeba_attacked").glob(
        "k[123]-*/checkpoint_vgg16.pth"
    )
)
CLASS_NAMES = ["Non-Blond_Hair", "Blond_Hair"]


def load_signature(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return checkpoint["experiment_signature"]


def build_test_dataset(signature: dict):
    dataset_cfg = signature["dataset"]
    classifier_cfg = signature["classifier"]
    dataset = get_celeba_attacked_dataset(
        data_paths=[str(DATA_ROOT)],
        normalize_data=dataset_cfg["normalize_data"],
        image_size=dataset_cfg["image_size"],
        attacked_classes=dataset_cfg["attacked_classes"],
        p_artifact=dataset_cfg["timestamp_probability_attacked"],
        artifact_type=dataset_cfg["artifact_type"],
        time_format=dataset_cfg["time_format"],
        num_concepts=dataset_cfg["num_concepts"],
        cooccurrence_probability=dataset_cfg["cooccurrence_probability"],
        entanglement_factor=dataset_cfg["entanglement_factor"],
        artifact_seed=dataset_cfg["artifact_seed"],
        brightness_factor=dataset_cfg["brightness_factor"],
        val_split=dataset_cfg["val_split"],
        test_split=dataset_cfg["test_split"],
        seed=dataset_cfg["split_seed"],
    )
    _, _, test_ids = dataset.do_train_val_test_split(
        classifier_cfg["val_split"],
        classifier_cfg["test_split"],
        seed=classifier_cfg["random_seed"],
    )
    return dataset.get_subset_by_idxs(test_ids)


def infer(checkpoint_path: Path, dataset, device: torch.device):
    model = get_vgg16(
        ckpt_path=str(checkpoint_path),
        pretrained=False,
        n_class=2,
    ).to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    all_logits = []
    all_targets = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu().long())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return probabilities, targets


def metric_rows(k: int, variant: str, model_kind: str, probabilities, targets):
    predictions = probabilities.argmax(axis=1)
    rows = []
    one_hot = np.eye(2, dtype=int)[targets]
    for class_id, class_name in enumerate(CLASS_NAMES):
        truth = one_hot[:, class_id]
        precision, recall, thresholds = precision_recall_curve(
            truth, probabilities[:, class_id]
        )
        f1_curve = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        best_index = int(np.argmax(f1_curve))
        best_threshold = (
            float(thresholds[best_index])
            if best_index < len(thresholds)
            else 1.0
        )
        row = {
            "k": k,
            "variant": variant,
            "model": model_kind,
            "class_id": class_id,
            "class_name": class_name,
            "n_test": int(len(targets)),
            "n_class": int(truth.sum()),
            "prevalence": float(truth.mean()),
            "average_precision": float(
                average_precision_score(truth, probabilities[:, class_id])
            ),
            "precision_at_default": float(
                precision_score(targets, predictions, labels=[class_id], average=None)[0]
            ),
            "recall_at_default": float(
                recall_score(targets, predictions, labels=[class_id], average=None)[0]
            ),
            "f1_at_default": float(
                f1_score(targets, predictions, labels=[class_id], average=None)[0]
            ),
            "best_f1": float(f1_curve[best_index]),
            "best_f1_threshold": best_threshold,
        }
        rows.append(row)
    return rows


def run() -> None:
    if len(CHECKPOINTS) != 3:
        raise RuntimeError(f"Expected exactly three K=1..3 checkpoints, found {CHECKPOINTS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}", flush=True)

    summary_rows = []
    run_rows = []
    subgroup_rows = []
    for checkpoint_path in CHECKPOINTS:
        signature = load_signature(checkpoint_path)
        k = int(signature["dataset"]["num_concepts"])
        variant = checkpoint_path.parent.name
        dataset = build_test_dataset(signature)
        targets_expected = dataset.metadata["targets"].to_numpy(dtype=int)
        print(f"K={k}: {variant}, n_test={len(dataset)}", flush=True)

        for model_kind, model_path in (
            ("clean_initialization", CLEAN_CHECKPOINT),
            ("attacked_trained", checkpoint_path),
        ):
            probabilities, targets = infer(model_path, dataset, device)
            if not np.array_equal(targets, targets_expected):
                raise RuntimeError("Inference target order does not match test metadata.")
            predictions = probabilities.argmax(axis=1)
            matrix = confusion_matrix(targets, predictions, labels=[0, 1])
            run_rows.append(
                {
                    "k": k,
                    "variant": variant,
                    "model": model_kind,
                    "n_test": int(len(targets)),
                    "accuracy": float(accuracy_score(targets, predictions)),
                    "macro_average_precision": float(
                        np.mean(
                            [
                                average_precision_score(
                                    (targets == class_id).astype(int),
                                    probabilities[:, class_id],
                                )
                                for class_id in (0, 1)
                            ]
                        )
                    ),
                    "macro_f1": float(f1_score(targets, predictions, average="macro")),
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                }
            )
            summary_rows.extend(
                metric_rows(k, variant, model_kind, probabilities, targets)
            )

            artifact_columns = signature["dataset"]["active_concepts"]
            any_artifact = (
                dataset.metadata[artifact_columns].to_numpy(dtype=int).any(axis=1)
            )
            timestamp = dataset.metadata["timestamp"].to_numpy(dtype=bool)
            for subgroup_name, subgroup_mask in (
                ("all", np.ones(len(targets), dtype=bool)),
                ("timestamp_present", timestamp),
                ("timestamp_absent", ~timestamp),
                ("any_artifact_present", any_artifact),
                ("no_artifact_present", ~any_artifact),
            ):
                subgroup_targets = targets[subgroup_mask]
                subgroup_predictions = predictions[subgroup_mask]
                if len(subgroup_targets) == 0:
                    continue
                subgroup_rows.append(
                    {
                        "k": k,
                        "variant": variant,
                        "model": model_kind,
                        "subgroup": subgroup_name,
                        "n": int(len(subgroup_targets)),
                        "positive_rate": float(subgroup_targets.mean()),
                        "accuracy": float(
                            accuracy_score(subgroup_targets, subgroup_predictions)
                        ),
                        "blond_precision": float(
                            precision_score(
                                subgroup_targets,
                                subgroup_predictions,
                                pos_label=1,
                                zero_division=0,
                            )
                        ),
                        "blond_recall": float(
                            recall_score(
                                subgroup_targets,
                                subgroup_predictions,
                                pos_label=1,
                                zero_division=0,
                            )
                        ),
                    }
                )
            print(
                f"  {model_kind}: accuracy={run_rows[-1]['accuracy']:.4f}, "
                f"macro_AP={run_rows[-1]['macro_average_precision']:.4f}",
                flush=True,
            )

    for filename, rows in (
        ("run_summary.csv", run_rows),
        ("per_class_metrics.csv", summary_rows),
        ("subgroup_metrics.csv", subgroup_rows),
    ):
        with (OUTPUT_DIR / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "device": str(device),
        "data_root": str(DATA_ROOT),
        "clean_checkpoint": str(CLEAN_CHECKPOINT),
        "runs": run_rows,
        "per_class": summary_rows,
        "subgroups": subgroup_rows,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
