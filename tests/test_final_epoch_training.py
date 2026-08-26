import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from experiments.utils.train_cavs import train_cavs, uses_final_epoch_checkpoint


class _TinyDataset:
    cache_identity = "tiny_cache"

    def get_concept_names(self):
        return ["a", "b"]

    def do_train_val_test_split(self, val_split, test_split, seed):
        return torch.tensor([0, 1]).numpy(), torch.tensor([2, 3]).numpy(), torch.tensor([], dtype=torch.long).numpy()


def _metrics(uniqueness):
    return {
        "auc_scores": [0.7, 0.7],
        "uniqueness": [uniqueness, uniqueness],
        "avg_precision": 0.7,
        "confusion_matrix": torch.zeros(2, 2, 2),
    }


class FinalEpochTrainingTest(unittest.TestCase):
    def _config(self, num_epochs, exit_criterion):
        return OmegaConf.create(
            {
                "dataset": {"name": "tiny", "_target_": "builtins.object"},
                "model": {"name": "unused"},
                "experiment": {"name": "concept_leakage", "out": "unused"},
                "train": {
                    "device": "cpu",
                    "random_seed": 42,
                    "val_ratio": 0.5,
                    "test_ratio": 0.0,
                    "batch_size": 2,
                    "num_workers": 0,
                    "learning_rate": 0.1,
                    "num_epochs": num_epochs,
                },
                "cav": {
                    "_target_": "cav_models.LogCAV",
                    "name": "log_cav",
                    "layer": "layer",
                    "cav_mode": "max",
                    "alpha": 0.0,
                    "beta": None,
                    "target_concepts": [],
                    "optimal_init": False,
                    "exit_criterion": exit_criterion,
                },
            }
        )

    def _run(self, num_epochs, exit_criterion, uniqueness_values):
        encodings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
        )
        labels = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        )
        saved = {}
        update_count = {"value": 0}

        def fake_train_epoch(dataloader, cav_model, weights, optimizer, device):
            update_count["value"] += 1
            with torch.no_grad():
                cav_model.weights.fill_(float(update_count["value"]))
            return 0.0, 0.0

        def fake_save_results(cavs, metrics, save_dir):
            saved["cavs"] = cavs.clone()

        metric_side_effect = [_metrics(value) for value in uniqueness_values]
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("experiments.utils.train_cavs.instantiate", return_value=_TinyDataset()),
                patch(
                    "experiments.utils.train_cavs.compute_cavs",
                    return_value=(torch.eye(2), torch.zeros(2)),
                ),
                patch("experiments.utils.train_cavs.train_epoch", side_effect=fake_train_epoch),
                patch(
                    "experiments.utils.train_cavs.eval_epoch",
                    side_effect=metric_side_effect,
                ),
                patch("experiments.utils.train_cavs.save_results", side_effect=fake_save_results),
                patch("experiments.utils.train_cavs.save_plots"),
            ):
                train_cavs(
                    self._config(num_epochs, exit_criterion),
                    encodings,
                    labels,
                    Path(tmpdir),
                )
        return update_count["value"], saved["cavs"]

    def test_null_exit_criterion_saves_state_after_exact_updates(self):
        updates, saved_cavs = self._run(3, None, [0.1, 0.2])
        self.assertEqual(updates, 3)
        torch.testing.assert_close(saved_cavs, torch.full((2, 2), 3.0))
        self.assertTrue(uses_final_epoch_checkpoint(None))

    def test_explicit_criterion_keeps_legacy_validation_selection(self):
        updates, saved_cavs = self._run(10, "orthogonality", [0.1, 0.2])
        self.assertEqual(updates, 11)
        torch.testing.assert_close(saved_cavs, torch.full((2, 2), 10.0))
        self.assertFalse(uses_final_epoch_checkpoint("orthogonality"))


if __name__ == "__main__":
    unittest.main()
