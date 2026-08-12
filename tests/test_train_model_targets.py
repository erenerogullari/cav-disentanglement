import unittest

import torch

from experiments.utils.train_model import _prepare_targets, _single_label_stats


class TrainModelTargetTest(unittest.TestCase):
    def test_scalar_class_targets_use_cross_entropy_shape_and_stats(self):
        targets = _prepare_targets(
            torch.tensor([1, 0]), torch.device("cpu"), single_label=True
        )
        self.assertEqual(tuple(targets.shape), (2,))
        self.assertEqual(targets.dtype, torch.long)

        logits = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
        stats = _single_label_stats(logits, targets)
        self.assertEqual(stats["accuracy"], 1.0)
        self.assertEqual(stats["precision"], 1.0)
        self.assertEqual(stats["recall"], 1.0)
        self.assertEqual(stats["f1"], 1.0)

    def test_multilabel_target_shape_remains_unchanged(self):
        targets = _prepare_targets(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.device("cpu"),
            single_label=False,
        )
        self.assertEqual(tuple(targets.shape), (2, 2))
        self.assertEqual(targets.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
