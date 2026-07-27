import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from hydra import compose, initialize_config_dir

from experiments.utils.activations import _get_features
from models import requires_lxt_localization
from utils.localization import get_localizations


class _ActivationAttribution:
    def __init__(self, activation):
        self.activation = activation

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(activations={"inspection_layer": self.activation})


class ViTLocalizationTest(unittest.TestCase):
    def test_capability_predicate_is_explicit(self):
        self.assertTrue(requires_lxt_localization("vit_b_16"))
        self.assertTrue(requires_lxt_localization("vit_b_32"))
        self.assertFalse(requires_lxt_localization("vit_l_16"))
        self.assertFalse(requires_lxt_localization("resnet18"))

    def test_two_dimensional_features_are_already_pooled(self):
        activation = torch.randn(3, 768)
        attribution = _ActivationAttribution(activation)
        batch = torch.randn(3, 3, 4, 4)

        for cav_mode in ("full", "max", "avg"):
            features = _get_features(
                batch.clone(),
                "inspection_layer",
                attribution,
                composite=None,
                cav_mode=cav_mode,
                device="cpu",
            )
            torch.testing.assert_close(features, activation)

    @patch("utils.localization.attribute_concept")
    def test_vit_dispatches_to_lxt(self, attribute_concept):
        expected = torch.ones(2, 8, 8)
        attribute_concept.return_value = expected

        attr, heatmaps = get_localizations(
            torch.randn(2, 3, 8, 8),
            torch.randn(16),
            attribution=None,
            composite=None,
            config={"layer_name": "inspection_layer"},
            device="cpu",
            model_name="vit_b_16",
            model=Mock(),
        )

        self.assertIsNone(attr)
        torch.testing.assert_close(heatmaps, expected)
        attribute_concept.assert_called_once()

    @patch("utils.localization.attribute_concept")
    @patch("utils.localization.get_features")
    def test_cnn_keeps_existing_attribution_path(
        self, get_features, attribute_concept
    ):
        x = torch.randn(2, 3, 8, 8)
        cav = torch.randn(4)
        get_features.return_value = torch.randn(2, 4, 2, 2)
        attr_result = SimpleNamespace(heatmap=torch.randn(2, 8, 8))
        attribution = Mock(return_value=attr_result)
        composite = object()

        attr, heatmaps = get_localizations(
            x,
            cav,
            attribution,
            composite,
            {"layer_name": "last_conv"},
            "cpu",
            model_name="resnet18",
            model=Mock(),
        )

        self.assertIs(attr, attr_result)
        torch.testing.assert_close(
            heatmaps, attr_result.heatmap.detach().cpu().clamp(min=0)
        )
        attribution.assert_called_once()
        attribute_concept.assert_not_called()

    def test_hydra_composes_both_vit_configs(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for model_name in ("vit_b_16", "vit_b_32"):
                cfg = compose(
                    config_name="concept_alignment",
                    overrides=[f"model={model_name}", "cav.alpha=0"],
                )
                self.assertEqual(cfg.model.name, model_name)
                self.assertEqual(cfg.model.n_class, 2)


if __name__ == "__main__":
    unittest.main()
