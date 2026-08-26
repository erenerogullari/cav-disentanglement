import unittest

import numpy as np
import pandas as pd
import torch

from experiments.concept_alignment.evaluate_alignment import (
    ALIGNMENT_PER_SAMPLE_COLUMNS,
)
from experiments.multi_concept_alignment.localization import (
    LOCALIZATION_PER_SAMPLE_COLUMNS,
    compute_localization_metrics,
    select_localization_sample_ids,
    summarize_localization_rows,
)


class _Dataset:
    idxs_test = np.arange(150, 350)
    sample_ids_by_artifact = {
        "timestamp": np.arange(100, 400),
        "brightness": np.arange(100, 400),
    }


class MultiConceptEvaluationTest(unittest.TestCase):
    def test_selection_is_deterministic_capped_positive_test_only(self):
        first = select_localization_sample_ids(_Dataset(), "timestamp", 128, 42)
        second = select_localization_sample_ids(_Dataset(), "timestamp", 128, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 128)
        self.assertTrue(set(first).issubset(set(_Dataset.idxs_test)))
        self.assertTrue(
            set(first).issubset(set(_Dataset.sample_ids_by_artifact["timestamp"]))
        )
        self.assertEqual(
            len(select_localization_sample_ids(_Dataset(), "brightness", 128, 42)),
            0,
        )

    def test_localization_metrics_on_synthetic_heatmap_and_mask(self):
        heatmaps = torch.zeros(1, 9, 9)
        masks = torch.zeros(1, 9, 9)
        heatmaps[0, 4, 4] = 4.0
        masks[0, 4, 4] = 1.0
        metrics = compute_localization_metrics(heatmaps, masks)
        torch.testing.assert_close(metrics["concept_relevance"], torch.ones(1))
        torch.testing.assert_close(
            metrics["mask_area_fraction"], torch.tensor([1.0 / 81.0])
        )
        self.assertGreaterEqual(float(metrics["otsu_iou"][0]), 0.0)
        self.assertLessEqual(float(metrics["otsu_iou"][0]), 1.0)

    def test_per_sample_and_summary_schemas(self):
        self.assertEqual(
            ALIGNMENT_PER_SAMPLE_COLUMNS,
            [
                "variant_id", "num_concepts", "concept_set", "cav_method",
                "alpha", "concept", "model", "cav_variant", "sample_id", "cosine",
            ],
        )
        row = {column: 0 for column in LOCALIZATION_PER_SAMPLE_COLUMNS}
        row.update(
            {
                "variant_id": "variant",
                "concept_set": '["timestamp"]',
                "cav_method": "pattern_cav",
                "cav_variant": "Baseline",
                "concept": "timestamp",
                "mask_variant": "raw",
                "num_concepts": 1,
                "alpha": 0.1,
            }
        )
        frame = pd.DataFrame([row, row], columns=LOCALIZATION_PER_SAMPLE_COLUMNS)
        summary = summarize_localization_rows(frame)
        self.assertEqual(int(summary.iloc[0]["n"]), 2)
        for metric in (
            "concept_relevance", "otsu_iou", "intersection_over_true_mask",
            "mask_area_fraction",
        ):
            self.assertIn(f"{metric}_mean", summary)
            self.assertIn(f"{metric}_std", summary)
            self.assertIn(f"{metric}_sem", summary)


if __name__ == "__main__":
    unittest.main()
