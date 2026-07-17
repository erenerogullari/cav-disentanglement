import unittest

import numpy as np
import torch

from experiments.model_correction.evaluate_heatmaps import (
    _build_cav_sets,
    _build_metric_plot_frames,
    _clean_cav,
    _clean_cavs,
    _store_metric_stats,
)


class ConceptCleaningTest(unittest.TestCase):
    def setUp(self):
        self.concept_names = ["timestamp", "brightness"]
        self.activations = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        self.labels = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        self.cavs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    def test_clean_cav_projects_out_mean_and_normalizes(self):
        non_concept_mean = torch.tensor([1.0, 0.0])

        cleaned = _clean_cav(torch.tensor([1.0, 1.0]), non_concept_mean)

        torch.testing.assert_close(cleaned, torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(float(cleaned.norm()), 1.0)
        self.assertAlmostEqual(float(torch.dot(cleaned, non_concept_mean)), 0.0)

    def test_zero_non_concept_mean_is_a_normalized_no_op(self):
        cleaned = _clean_cav(
            torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.0])
        )

        torch.testing.assert_close(cleaned, torch.tensor([0.6, 0.8]))

    def test_clean_cavs_uses_a_separate_negative_pool_per_concept(self):
        cleaned = _clean_cavs(
            self.cavs,
            self.activations,
            self.labels,
            self.concept_names,
            self.concept_names,
        )

        torch.testing.assert_close(cleaned[0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(cleaned[1], torch.tensor([1.0, 0.0]))

    def test_cleaning_rejects_missing_negatives_and_collapsed_directions(self):
        with self.assertRaisesRegex(ValueError, "No non-concept samples"):
            _clean_cavs(
                self.cavs,
                self.activations,
                torch.ones_like(self.labels),
                self.concept_names,
                ["timestamp"],
            )

        with self.assertRaisesRegex(ValueError, "collapsed"):
            _clean_cav(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))

    def test_cleaning_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            _clean_cav(torch.ones(2), torch.ones(3))

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            _clean_cavs(
                self.cavs,
                self.activations[:2],
                self.labels,
                self.concept_names,
                ["timestamp"],
            )

        with self.assertRaisesRegex(ValueError, "same number of features"):
            _clean_cavs(
                self.cavs,
                torch.ones(4, 3),
                self.labels,
                self.concept_names,
                ["timestamp"],
            )

    def test_cav_sets_preserve_existing_outputs_and_add_cleaned_variants(self):
        original = _build_cav_sets(
            self.cavs,
            self.cavs,
            self.concept_names,
            self.concept_names,
            cleaning_enabled=False,
        )
        self.assertEqual(list(original), ["Baseline", "Orthogonal"])

        extended = _build_cav_sets(
            self.cavs,
            self.cavs,
            self.concept_names,
            self.concept_names,
            cleaning_enabled=True,
            activations=self.activations,
            labels=self.labels,
        )
        self.assertEqual(
            list(extended),
            ["Baseline", "Orthogonal", "Baseline Cleaned", "Orthogonal Cleaned"],
        )
        torch.testing.assert_close(extended["Baseline"], self.cavs)
        torch.testing.assert_close(extended["Orthogonal"], self.cavs)

    def test_cleaning_requires_preprocessing_data(self):
        with self.assertRaisesRegex(ValueError, "requires preprocessing"):
            _build_cav_sets(
                self.cavs,
                self.cavs,
                self.concept_names,
                self.concept_names,
                cleaning_enabled=True,
            )

    def test_metric_frames_keep_all_requested_cav_series(self):
        cav_order = [
            "Baseline",
            "Orthogonal",
            "Baseline Cleaned",
            "Orthogonal Cleaned",
        ]
        results = {}
        for index, cav_name in enumerate(cav_order):
            _store_metric_stats(
                results,
                "iou",
                "timestamp",
                cav_name,
                np.array([index, index + 1], dtype=float),
            )

        data, data_std = _build_metric_plot_frames(
            results, "iou", "timestamp", "IoU", cav_order
        )

        self.assertEqual(data["CAV"].tolist(), cav_order)
        self.assertEqual(data_std["CAV"].tolist(), cav_order)
        self.assertEqual(len(results), 2 * len(cav_order))


if __name__ == "__main__":
    unittest.main()
