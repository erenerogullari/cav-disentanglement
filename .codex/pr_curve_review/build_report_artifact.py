"""Build a validated-artifact input for the K=1..3 VGG PR review."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
TITLE = "VGG precision–recall review for K=1–3"
GENERATED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open(newline="") as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def values_query(rows: list[dict], columns: list[str], order_by: str) -> str:
    values = ",\n    ".join(
        "(" + ", ".join(sql_literal(row[column]) for column in columns) + ")"
        for row in rows
    )
    query = (
        "WITH reviewed_metrics(" + ", ".join(columns) + ") AS (\n"
        "  VALUES\n    " + values + "\n)\n"
        "SELECT " + ", ".join(columns) + "\n"
        "FROM reviewed_metrics\n"
        f"ORDER BY {order_by};"
    )
    connection = sqlite3.connect(":memory:")
    try:
        returned = connection.execute(query).fetchall()
    finally:
        connection.close()
    if len(returned) != len(rows):
        raise RuntimeError("Report-source SQL did not return all reviewed rows.")
    return query


def build_rows():
    run_rows = read_csv("run_summary.csv")
    class_rows = read_csv("per_class_metrics.csv")
    subgroup_rows = read_csv("subgroup_metrics.csv")
    history_rows = read_csv("training_history.csv")
    best_rows = {int(row["k"]): row for row in read_csv("best_epochs.csv")}

    comparison_rows = []
    ap_rows = []
    timestamp_rows = []
    for k in (1, 2, 3):
        trained_run = next(
            row
            for row in run_rows
            if int(row["k"]) == k and row["model"] == "attacked_trained"
        )
        clean_run = next(
            row
            for row in run_rows
            if int(row["k"]) == k and row["model"] == "clean_initialization"
        )
        trained_classes = {
            int(row["class_id"]): row
            for row in class_rows
            if int(row["k"]) == k and row["model"] == "attacked_trained"
        }
        concept_names = {
            1: "timestamp",
            2: "timestamp + box",
            3: "timestamp + box + brightness",
        }
        comparison_rows.append(
            {
                "k": k,
                "concepts": concept_names[k],
                "n_test": int(trained_run["n_test"]),
                "accuracy": as_float(trained_run, "accuracy"),
                "accuracy_delta_pp": 100
                * (as_float(trained_run, "accuracy") - as_float(clean_run, "accuracy")),
                "macro_ap": as_float(trained_run, "macro_average_precision"),
                "macro_ap_delta_pp": 100
                * (
                    as_float(trained_run, "macro_average_precision")
                    - as_float(clean_run, "macro_average_precision")
                ),
                "blond_ap": as_float(trained_classes[1], "average_precision"),
                "blond_precision": as_float(
                    trained_classes[1], "precision_at_default"
                ),
                "blond_recall": as_float(trained_classes[1], "recall_at_default"),
                "blond_f1": as_float(trained_classes[1], "f1_at_default"),
                "best_epoch": int(best_rows[k]["best_epoch"]),
                "best_val_accuracy": float(best_rows[k]["best_val_accuracy"]),
            }
        )
        for class_id, class_name in ((0, "Non-Blond Hair"), (1, "Blond Hair")):
            class_row = trained_classes[class_id]
            ap_rows.append(
                {
                    "k_label": f"K={k}",
                    "k": k,
                    "class_name": class_name,
                    "average_precision": as_float(class_row, "average_precision"),
                    "prevalence": as_float(class_row, "prevalence"),
                    "n_class": int(class_row["n_class"]),
                    "n_test": int(class_row["n_test"]),
                }
            )

        trained_subgroups = {
            row["subgroup"]: row
            for row in subgroup_rows
            if int(row["k"]) == k and row["model"] == "attacked_trained"
        }
        clean_subgroups = {
            row["subgroup"]: row
            for row in subgroup_rows
            if int(row["k"]) == k and row["model"] == "clean_initialization"
        }
        timestamp_rows.append(
            {
                "k": k,
                "timestamp_present_n": int(trained_subgroups["timestamp_present"]["n"]),
                "timestamp_present_positive_rate": as_float(
                    trained_subgroups["timestamp_present"], "positive_rate"
                ),
                "trained_recall_timestamp_present": as_float(
                    trained_subgroups["timestamp_present"], "blond_recall"
                ),
                "trained_recall_timestamp_absent": as_float(
                    trained_subgroups["timestamp_absent"], "blond_recall"
                ),
                "clean_recall_timestamp_absent": as_float(
                    clean_subgroups["timestamp_absent"], "blond_recall"
                ),
            }
        )

    training_rows = [
        {
            "k_label": f"K={int(row['k'])}",
            "k": int(row["k"]),
            "epoch": int(row["epoch"]),
            "train_loss": as_float(row, "train_loss"),
            "train_accuracy": as_float(row, "train_accuracy"),
            "val_loss": as_float(row, "val_loss"),
            "val_accuracy": as_float(row, "val_accuracy"),
            "val_f1": as_float(row, "val_f1"),
        }
        for row in history_rows
    ]
    return comparison_rows, ap_rows, training_rows, timestamp_rows


def build_artifact() -> dict:
    comparison_rows, ap_rows, training_rows, timestamp_rows = build_rows()
    comparison_columns = list(comparison_rows[0])
    ap_columns = list(ap_rows[0])
    training_columns = list(training_rows[0])
    timestamp_columns = list(timestamp_rows[0])

    comparison_sql = values_query(comparison_rows, comparison_columns, "k")
    ap_sql = values_query(ap_rows, ap_columns, "k, class_name")
    training_sql = values_query(training_rows, training_columns, "k, epoch")
    timestamp_sql = values_query(timestamp_rows, timestamp_columns, "k")

    sources = [
        {
            "id": "recomputed_checkpoint_metrics",
            "label": "Recomputed K=1–3 checkpoint test metrics",
            "query": {
                "engine": "SQLite",
                "language": "SQL",
                "executed_at": GENERATED_AT,
                "sql": comparison_sql,
                "description": "Selects the reviewed checkpoint comparison rows produced by deterministic test inference.",
                "tables_used": ["reviewed_metrics"],
                "filters": [
                    "Deterministic CelebA test split, seed 42",
                    "K in 1, 2, 3",
                    "2,115 test samples per K",
                    "Argmax operating point for precision, recall, F1, and accuracy",
                ],
                "metric_definitions": [
                    "Average precision is sklearn average_precision_score over the class softmax probability and one-vs-rest class target.",
                    "Macro AP is the unweighted mean of Non-Blond Hair AP and Blond Hair AP.",
                    "Accuracy delta is trained attacked-checkpoint accuracy minus clean-initialization accuracy on the same attacked test cohort, in percentage points.",
                ],
            },
        },
        {
            "id": "per_class_ap_metrics",
            "label": "Per-class average precision and prevalence",
            "query": {
                "engine": "SQLite",
                "language": "SQL",
                "executed_at": GENERATED_AT,
                "sql": ap_sql,
                "description": "Selects trained-checkpoint AP and class prevalence by K and one-vs-rest class.",
                "tables_used": ["reviewed_metrics"],
                "filters": [
                    "Attacked-trained checkpoint only",
                    "Deterministic 2,115-sample test split",
                ],
                "metric_definitions": [
                    "Average precision summarizes the precision–recall ranking curve; the no-skill reference equals class prevalence.",
                ],
            },
        },
        {
            "id": "classifier_training_history",
            "label": "K=1–3 classifier epoch histories",
            "query": {
                "engine": "SQLite",
                "language": "SQL",
                "executed_at": GENERATED_AT,
                "sql": training_sql,
                "description": "Selects the 20 logged train/validation epochs for each K.",
                "tables_used": ["reviewed_metrics"],
                "filters": ["All 20 epochs for K=1, K=2, and K=3"],
                "metric_definitions": [
                    "Validation accuracy is whole-split two-class accuracy at each epoch.",
                    "The persisted checkpoint is selected by maximum validation accuracy.",
                ],
            },
        },
        {
            "id": "timestamp_subgroup_metrics",
            "label": "Timestamp-present and timestamp-absent subgroup metrics",
            "query": {
                "engine": "SQLite",
                "language": "SQL",
                "executed_at": GENERATED_AT,
                "sql": timestamp_sql,
                "description": "Selects observational test metrics split by timestamp presence.",
                "tables_used": ["reviewed_metrics"],
                "filters": [
                    "Same deterministic test cohort as overall metrics",
                    "Timestamp presence generated with 0.4 probability for Blond Hair and 0.005 otherwise",
                ],
                "metric_definitions": [
                    "Blond recall is true Blond Hair predictions divided by all Blond Hair examples within the indicated timestamp subgroup.",
                ],
            },
        },
        {
            "id": "analysis_code",
            "label": "Recomputation code",
            "path": ".codex/pr_curve_review/analyze_pr.py",
        },
        {
            "id": "training_log_parser",
            "label": "Training-log parser",
            "path": ".codex/pr_curve_review/parse_training_logs.py",
        },
        {
            "id": "original_pr_plots",
            "label": "Original checkpoint precision–recall PDFs",
            "path": "checkpoints/celeba_attacked",
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Technical validation of K=1–3 CelebA VGG16 precision–recall behavior.",
            "generatedAt": GENERATED_AT,
            "filters": [],
            "cards": [],
            "charts": [
                {
                    "id": "ap_by_k_and_class",
                    "title": "Per-class average precision by K",
                    "subtitle": "Attacked-trained checkpoints on the same 2,115-sample test split; higher is better",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "per_class_ap",
                    "sourceId": "per_class_ap_metrics",
                    "encodings": {
                        "x": {"field": "k_label", "type": "ordinal", "label": "Concept count"},
                        "y": {"field": "average_precision", "type": "quantitative", "format": "percent", "label": "Average precision"},
                        "color": {"field": "class_name", "type": "nominal", "label": "Class"},
                        "tooltip": [
                            {"field": "average_precision", "type": "quantitative", "format": "percent", "label": "Average precision"},
                            {"field": "prevalence", "type": "quantitative", "format": "percent", "label": "Prevalence baseline"},
                            {"field": "n_class", "type": "quantitative", "format": "number", "label": "Class samples"},
                        ],
                    },
                    "xAxisTitle": "Nested concept set",
                    "yAxisTitle": "Average precision",
                    "valueFormat": "percent",
                    "layout": "full",
                    "maxRows": 6,
                },
                {
                    "id": "validation_accuracy_history",
                    "title": "Validation accuracy over classifier training",
                    "subtitle": "Twenty epochs per K; persisted checkpoints were selected by peak validation accuracy",
                    "showDescription": True,
                    "type": "line",
                    "dataset": "training_history",
                    "sourceId": "classifier_training_history",
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                        "y": {"field": "val_accuracy", "type": "quantitative", "format": "percent", "label": "Validation accuracy"},
                        "color": {"field": "k_label", "type": "nominal", "label": "Concept count"},
                        "tooltip": [
                            {"field": "val_accuracy", "type": "quantitative", "format": "percent", "label": "Validation accuracy"},
                            {"field": "val_f1", "type": "quantitative", "format": "percent", "label": "Validation macro F1"},
                            {"field": "val_loss", "type": "quantitative", "format": "number", "label": "Validation loss"},
                        ],
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Validation accuracy",
                    "valueFormat": "percent",
                    "layout": "full",
                    "maxRows": 60,
                },
            ],
            "tables": [
                {
                    "id": "checkpoint_comparison",
                    "title": "Checkpoint test metrics",
                    "subtitle": "Exact metrics for the attacked-trained VGG16 checkpoints; deltas compare with their shared clean initialization",
                    "showDescription": True,
                    "dataset": "checkpoint_comparison",
                    "density": "spacious",
                    "sourceId": "recomputed_checkpoint_metrics",
                    "defaultSort": {"field": "k", "direction": "asc"},
                    "columns": [
                        {"field": "k", "label": "K", "type": "number", "format": "number"},
                        {"field": "concepts", "label": "Concepts", "type": "text"},
                        {"field": "accuracy", "label": "Accuracy", "type": "percent", "format": "percent"},
                        {"field": "accuracy_delta_pp", "label": "Accuracy Δ vs clean", "type": "number", "format": "number", "unit": "pp", "movement": True},
                        {"field": "macro_ap", "label": "Macro AP", "type": "percent", "format": "percent"},
                        {"field": "macro_ap_delta_pp", "label": "Macro AP Δ vs clean", "type": "number", "format": "number", "unit": "pp", "movement": True},
                        {"field": "blond_ap", "label": "Blond AP", "type": "percent", "format": "percent"},
                        {"field": "blond_precision", "label": "Blond precision", "type": "percent", "format": "percent"},
                        {"field": "blond_recall", "label": "Blond recall", "type": "percent", "format": "percent"},
                        {"field": "blond_f1", "label": "Blond F1", "type": "percent", "format": "percent"},
                        {"field": "best_epoch", "label": "Best epoch", "type": "number", "format": "number"},
                    ],
                },
                {
                    "id": "timestamp_subgroups",
                    "title": "Blond-Hair recall by timestamp presence",
                    "subtitle": "Observational subgroup comparison; timestamp-positive and timestamp-negative cohorts have very different class prevalence",
                    "showDescription": True,
                    "dataset": "timestamp_subgroups",
                    "density": "spacious",
                    "sourceId": "timestamp_subgroup_metrics",
                    "defaultSort": {"field": "k", "direction": "asc"},
                    "columns": [
                        {"field": "k", "label": "K", "type": "number", "format": "number"},
                        {"field": "timestamp_present_n", "label": "Timestamp-present n", "type": "number", "format": "number"},
                        {"field": "timestamp_present_positive_rate", "label": "Timestamp-present Blond rate", "type": "percent", "format": "percent"},
                        {"field": "trained_recall_timestamp_present", "label": "Trained recall: timestamp present", "type": "percent", "format": "percent"},
                        {"field": "trained_recall_timestamp_absent", "label": "Trained recall: timestamp absent", "type": "percent", "format": "percent"},
                        {"field": "clean_recall_timestamp_absent", "label": "Clean recall: timestamp absent", "type": "percent", "format": "percent"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "recomputed_checkpoint_metrics",
                    "body": "## Technical summary\n\n**Overall assessment: the three checkpoints successfully optimize the Blond-Hair classification objective, and increasing K from 1 to 3 does not cause a visible precision–recall collapse.** Blond-Hair average precision is 91.7%, 91.6%, and 92.2% for K=1, K=2, and K=3, far above the 19.1% class-prevalence reference. Test accuracy stays between 94.0% and 94.3%.\n\n**The stronger claim that fine-tuning learned a substantially better classifier is not supported.** The shared clean initialization already reaches 93.7% test accuracy and roughly 95.6%–95.7% macro AP on the attacked cohorts. Fine-tuning adds only 0.28–0.61 percentage points of accuracy, while macro AP changes by −0.33, −0.26, and +0.15 points. The result is best described as retention and operating-point adjustment of an already learned objective, not learning from scratch.",
                },
                {
                    "id": "pr_finding",
                    "type": "markdown",
                    "sourceId": "per_class_ap_metrics",
                    "body": "## Precision–recall quality is high and stable across K\n\nThe minority Blond-Hair class is the decisive curve: its AP remains above 91.6% for every checkpoint despite only 405 positive examples in the 2,115-sample test set. Non-Blond-Hair AP remains above 99.0%. K=3 is marginally highest on Blond-Hair AP, so the observed sequence is not a monotonic degradation with more entangled concepts. Differences below one percentage point should not be treated as meaningful without repeated seeds or paired uncertainty estimates.",
                },
                {"id": "ap_chart", "type": "chart", "chartId": "ap_by_k_and_class", "layout": "full"},
                {
                    "id": "operating_point_finding",
                    "type": "markdown",
                    "sourceId": "recomputed_checkpoint_metrics",
                    "body": "## The default decision point is usable but more conservative after fine-tuning\n\nAt the default argmax decision point, Blond-Hair precision is 84.7%–86.3% and recall is 83.7%–84.4%. Compared with the clean initialization, fine-tuning generally raises minority-class precision while lowering recall. This explains why accuracy can improve slightly even when ranking quality, measured by AP, is flat or lower. K=2 has the best test accuracy (94.33%) and macro F1 (90.73%), but none of the three K settings is materially separated by the available single-run evidence.",
                },
                {"id": "metrics_table", "type": "table", "tableId": "checkpoint_comparison", "layout": "full"},
                {
                    "id": "convergence_finding",
                    "type": "markdown",
                    "sourceId": "classifier_training_history",
                    "body": "## Optimization converged, with early best epochs for K=2 and K=3\n\nRelative to epoch 1, selected validation accuracy improved by 6.62 points for K=1, 10.22 points for K=2, and 1.89 points for K=3. The saved best epochs were 16, 4, and 8 respectively. Later epochs fluctuate rather than improve monotonically, so loading the peak-validation checkpoint was necessary. Because selection uses validation accuracy—not AP or minority-class F1—the saved operating point is expected to favor overall correctness on this imbalanced task.",
                },
                {"id": "training_chart", "type": "chart", "chartId": "validation_accuracy_history", "layout": "full"},
                {
                    "id": "shortcut_finding",
                    "type": "markdown",
                    "sourceId": "timestamp_subgroup_metrics",
                    "body": "## Timestamp-absent recall is the main warning signal\n\nFor the trained checkpoints, Blond-Hair recall is 95.5% when timestamp is present but only 77.6%, 76.8%, and 76.4% when it is absent. The clean initialization reaches about 90.8% recall in the timestamp-absent subgroup. This is consistent with increased use of the correlated artifact, but it is not a causal shortcut estimate: timestamp presence changes class prevalence dramatically (91.7% Blond among timestamp-present samples), and these are not paired counterfactual cohorts.",
                },
                {"id": "subgroup_table", "type": "table", "tableId": "timestamp_subgroups", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": "## Scope, data, and metric definitions\n\nAll K values use the same deterministic CelebA split (seed 42), 2,115 test examples, the same clean VGG16 initialization, 20 training epochs, and the same optimizer settings. Only the nested artifact set changes: timestamp; timestamp + box; timestamp + box + brightness.\n\nAverage precision summarizes the one-vs-rest precision–recall ranking using softmax probabilities. Its no-skill reference is the class prevalence: 19.1% for Blond Hair and 80.9% for Non-Blond Hair. Precision, recall, F1, and accuracy use the default argmax decision rule, equivalent to a 0.5 threshold in this two-class softmax model.",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "analysis_code",
                    "body": "## Methodology and reproducibility\n\nEach saved checkpoint was signature-checked, reloaded in the existing project environment, and evaluated against its reconstructed deterministic attacked test dataset. The clean checkpoint was evaluated on each identical K-specific cohort as a baseline. Class-level AP, threshold metrics, confusion counts, and artifact subgroups were recomputed independently of the saved PDFs. Training histories and selected epochs were parsed from the corresponding Hydra classifier logs.",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Limitations and visualization issues\n\nThe original PDFs omit class legends, AP values, and prevalence baselines; the scatter plots contain two unlabeled points. They are directionally correct but not self-interpreting. The checkpoints also save model weights and signatures, not raw test predictions or epoch histories, so uncertainty intervals are absent. Results come from one seed and cannot establish that sub-percentage-point K differences are reproducible. Finally, endpoint PR curves establish predictive quality, not whether the classifier causally relies on each artifact.",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## Recommended next steps\n\n1. Add class labels, AP values, and prevalence reference lines to the PR PDFs.\n2. Save raw test targets/probabilities and epoch histories beside every classifier checkpoint.\n3. Select or at least report checkpoints by Blond-Hair AP/F1 in addition to overall validation accuracy.\n4. Run paired counterfactual evaluation: score each test image clean and with one artifact toggled while holding identity fixed.\n5. Repeat K=1–3 with multiple classifier seeds and report paired bootstrap intervals before interpreting small differences.",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## Further questions\n\nWould the same stability hold for K=4 and K=5, where checkerboard and watermark add localized cues? Does increased timestamp-absent error persist under paired artifact removal? And do classifier shortcut measures track the downstream CAV alignment and localization changes that motivate the sweep?",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "checkpoint_comparison": comparison_rows,
                "per_class_ap": ap_rows,
                "training_history": training_rows,
                "timestamp_subgroups": timestamp_rows,
            },
            "accessIssues": [],
        },
        "sources": [],
        "package_info": {
            "analysis_as_of": "18 August 2026",
            "audience": "technical",
            "delivery_mode": "portable_html",
        },
    }


if __name__ == "__main__":
    artifact = build_artifact()
    (OUTPUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False)
    )
