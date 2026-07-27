import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from experiments.utils.activations import extract_latents


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, labels: torch.Tensor):
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return torch.tensor([float(index)]), torch.tensor(0)

    def get_labels(self):
        return self.labels


class ActivationCacheTest(unittest.TestCase):
    def _config(self):
        return OmegaConf.create(
            {
                "dataset": {"name": "dataset"},
                "model": {"name": "model"},
                "cav": {"layer": "layer", "cav_mode": "max"},
                "train": {"batch_size": 2, "num_workers": 0, "device": "cpu"},
            }
        )

    def test_incompatible_cached_labels_trigger_latent_reextraction(self):
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        dataset = _Dataset(labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "variables/dataset/model/layer.pth"
            cache_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "encs": torch.full((2, 4), -1.0),
                    "labels": torch.zeros((2, 2)),
                },
                cache_path,
            )

            fresh_encodings = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            with (
                patch(
                    "experiments.utils.activations.get_original_cwd",
                    return_value=tmpdir,
                ),
                patch("experiments.utils.activations.get_canonizer", return_value=[]),
                patch("experiments.utils.activations.EpsilonPlusFlat"),
                patch("experiments.utils.activations.CondAttribution"),
                patch(
                    "experiments.utils.activations._get_features",
                    return_value=fresh_encodings,
                ) as get_features,
            ):
                encodings, actual_labels = extract_latents(
                    self._config(), torch.nn.Identity(), dataset
                )

            get_features.assert_called_once()
            torch.testing.assert_close(encodings, fresh_encodings)
            torch.testing.assert_close(actual_labels, labels)

            cached = torch.load(cache_path, weights_only=True)
            torch.testing.assert_close(cached["encs"], fresh_encodings)
            torch.testing.assert_close(cached["labels"], labels)

    def test_matching_cache_is_reused(self):
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        encodings = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        dataset = _Dataset(labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "variables/dataset/model/layer.pth"
            cache_path.parent.mkdir(parents=True)
            torch.save({"encs": encodings, "labels": labels}, cache_path)

            with (
                patch(
                    "experiments.utils.activations.get_original_cwd",
                    return_value=tmpdir,
                ),
                patch("experiments.utils.activations._get_features") as get_features,
            ):
                actual_encodings, actual_labels = extract_latents(
                    self._config(), torch.nn.Identity(), dataset
                )

            get_features.assert_not_called()
            torch.testing.assert_close(actual_encodings, encodings)
            torch.testing.assert_close(actual_labels, labels)


if __name__ == "__main__":
    unittest.main()
