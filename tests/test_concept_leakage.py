import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from experiments.concept_leakage.evaluate import (
    _build_leakage_tables,
    compute_directed_leakage,
    compute_localization_metrics,
    evaluate_concept_leakage,
)


class _LeakageDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.sample_ids_by_concept = {
            "source": [0, 1],
            "distractor": [0, 2],
        }
        source = torch.zeros(4, 4, dtype=torch.bool)
        source[:2, :2] = True
        distractor = torch.zeros(4, 4, dtype=torch.bool)
        distractor[2:, 2:] = True
        self.masks = {"source": source, "distractor": distractor}
        self.mean = torch.zeros(3)
        self.var = torch.ones(3)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index) / 10), self.labels[index]

    def get_labels(self):
        return self.labels.clone()

    def get_concept_names(self):
        return ["source", "distractor"]

    def get_concept_masks(self, index):
        return {name: mask.clone() for name, mask in self.masks.items()}

    def get_sample_name(self, index):
        return str(100 + index)

    def reverse_normalization(self, image):
        return image * 255


class ConceptLeakageMetricTest(unittest.TestCase):
    def test_target_metrics_include_pointing_iou_and_zero_detection(self):
        heatmap = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        mask = torch.zeros(3, 3, dtype=torch.bool)
        mask[:2, :2] = True
        metrics = compute_localization_metrics(heatmap, mask)
        self.assertAlmostEqual(metrics["target_relevance"], 1.0)
        self.assertEqual(metrics["pointing_game"], 1)
        self.assertGreaterEqual(metrics["iou"], 0.0)
        self.assertGreaterEqual(metrics["intersection_over_gt"], 0.0)
        self.assertEqual(metrics["zero_heatmap"], 0)

        zero = compute_localization_metrics(torch.zeros(3, 3), mask)
        self.assertEqual(zero["zero_heatmap"], 1)
        self.assertEqual(zero["pointing_game"], 0)

    def test_directed_leakage_excludes_source_overlap(self):
        heatmap = torch.ones(2, 2)
        source = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
        distractor = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
        leakage, area = compute_directed_leakage(heatmap, source, distractor)
        self.assertAlmostEqual(leakage, 0.25)
        self.assertEqual(area, 1)

    def test_missing_pair_support_is_nan_with_zero_count(self):
        frame = pd.DataFrame(
            [
                {
                    "source_concept": "a",
                    "mask_concept": "a",
                    "leakage": 0.8,
                }
            ]
        )
        matrix, counts = _build_leakage_tables(frame, ["a", "b"])
        self.assertAlmostEqual(matrix.loc["a", "a"], 0.8)
        self.assertTrue(pd.isna(matrix.loc["a", "b"]))
        self.assertEqual(counts.loc["a", "b"], 0)

    def test_mocked_evaluator_writes_all_artifacts(self):
        cfg = OmegaConf.create(
            {
                "model": {"name": "vgg16"},
                "cav": {"layer": "features.0", "cav_mode": "max"},
                "evaluation": {
                    "device": "cpu",
                    "batch_size": 2,
                    "max_samples_per_concept": 2,
                    "overlays_per_concept": 1,
                    "random_seed": 42,
                },
            }
        )

        def localization_fn(cav, images, *args, **kwargs):
            heatmaps = torch.zeros(len(images), 4, 4)
            heatmaps[:, :2, :2] = 2 if cav[0, 0] > 0 else 1
            heatmaps[:, 2:, 2:] = 1
            return heatmaps

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            evaluate_concept_leakage(
                cfg,
                torch.nn.Identity(),
                _LeakageDataset(),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                save_dir,
                localization_fn=localization_fn,
            )
            for name in (
                "localization_per_sample.csv",
                "localization_summary.csv",
                "leakage_per_sample.csv",
                "leakage_matrix.csv",
                "leakage_counts.csv",
            ):
                self.assertTrue((save_dir / "metrics" / name).is_file(), name)
            self.assertTrue((save_dir / "media" / "leakage_matrix.png").is_file())
            self.assertEqual(
                len(list((save_dir / "media" / "localization").glob("*.png"))),
                2,
            )


if __name__ == "__main__":
    unittest.main()
