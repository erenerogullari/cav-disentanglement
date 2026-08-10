import unittest

import torch

from experiments.model_correction.evaluate_heatmaps import _dilate_binary_masks
from utils.localization import compute_concept_relevance


class ConceptRelevanceMetricTest(unittest.TestCase):
    def test_single_heatmap_fraction_inside_mask(self):
        heatmap = torch.ones(2, 2)
        mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

        score = compute_concept_relevance(heatmap, mask)

        torch.testing.assert_close(score, torch.tensor(0.25))

    def test_negative_relevance_is_excluded(self):
        heatmap = torch.tensor([[2.0, -10.0], [1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        score = compute_concept_relevance(heatmap, mask)

        torch.testing.assert_close(score, torch.tensor(0.5))

    def test_zero_heatmap_is_safe(self):
        score = compute_concept_relevance(torch.zeros(2, 2), torch.ones(2, 2))

        torch.testing.assert_close(score, torch.tensor(0.0))

    def test_batch_mean_matches_direct_aggregate_formula(self):
        heatmaps = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        )
        masks = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        )

        per_sample = compute_concept_relevance(heatmaps, masks)
        direct = (heatmaps * masks).sum((1, 2)) / (
            heatmaps.sum((1, 2)) + 1e-10
        )

        torch.testing.assert_close(per_sample.mean(), direct.mean())

    def test_three_pixel_box_dilation_uses_seven_by_seven_neighborhood(self):
        mask = torch.zeros(1, 9, 9)
        mask[0, 4, 4] = 1

        dilated = _dilate_binary_masks(mask, padding=3)

        self.assertEqual(int(dilated.sum()), 49)
        self.assertEqual(float(dilated[0, 1, 1]), 1.0)
        self.assertEqual(float(dilated[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
