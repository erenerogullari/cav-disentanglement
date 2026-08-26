"""Extract K=1..3 VGG training histories from Hydra logs."""

from __future__ import annotations

import csv
import re
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOGS = {
    1: REPOSITORY_ROOT
    / "outputs/2026-08-17/12-47-10/run_multi_concept_alignment.log",
    2: REPOSITORY_ROOT
    / "outputs/2026-08-17/18-52-39/run_multi_concept_alignment.log",
    3: REPOSITORY_ROOT
    / "outputs/2026-08-18/05-27-47/run_multi_concept_alignment.log",
}
METRICS = r"Loss=(?P<loss>[0-9.]+) Acc=(?P<accuracy>[0-9.]+) F1=(?P<f1>[0-9.]+) Precision=(?P<precision>[0-9.]+) Recall=(?P<recall>[0-9.]+)"
TRAIN_PATTERN = re.compile(r"Epoch\s+(?P<epoch>\d+) \| Train stats:\s+" + METRICS)
VAL_PATTERN = re.compile(r"\| Val stats:\s+" + METRICS)
BEST_PATTERN = re.compile(r"best model from epoch (?P<epoch>\d+) with val_acc=(?P<accuracy>[0-9.]+)")


def run() -> None:
    rows = []
    best_rows = []
    for k, log_path in LOGS.items():
        pending_epoch = None
        parsed_by_epoch = {}
        for line in log_path.read_text().splitlines():
            train_match = TRAIN_PATTERN.search(line)
            if train_match:
                values = train_match.groupdict()
                pending_epoch = int(values.pop("epoch"))
                parsed_by_epoch[pending_epoch] = {
                    "k": k,
                    "epoch": pending_epoch,
                    **{f"train_{key}": float(value) for key, value in values.items()},
                }
                continue
            val_match = VAL_PATTERN.search(line)
            if val_match and pending_epoch is not None:
                parsed_by_epoch[pending_epoch].update(
                    {
                        f"val_{key}": float(value)
                        for key, value in val_match.groupdict().items()
                    }
                )
                pending_epoch = None
                continue
            best_match = BEST_PATTERN.search(line)
            if best_match:
                best_rows.append(
                    {
                        "k": k,
                        "best_epoch": int(best_match.group("epoch")),
                        "best_val_accuracy": float(best_match.group("accuracy")),
                    }
                )
        if len(parsed_by_epoch) != 20:
            raise RuntimeError(f"Expected 20 epochs for K={k}, found {len(parsed_by_epoch)}")
        rows.extend(parsed_by_epoch.values())

    with (OUTPUT_DIR / "training_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "best_epochs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)


if __name__ == "__main__":
    run()
