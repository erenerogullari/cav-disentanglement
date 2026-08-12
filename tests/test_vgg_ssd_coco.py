import unittest
from unittest.mock import patch

import torch
from crp.attribution import CondAttribution
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from zennit.composites import EpsilonPlusFlat

from experiments.utils.activations import _get_features
from experiments.utils.localization import get_localization
from models import get_canonizer
from models.vgg import get_vgg16_ssd_coco


class VGG16SSDCOCOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ssd300_vgg16(weights=None, weights_backbone=None)

    def test_loader_selects_coco_weights_and_preserves_vgg_indices(self):
        with patch(
            "models.vgg.ssd300_vgg16", return_value=self.detector
        ) as detector_loader:
            model = get_vgg16_ssd_coco(pretrained=True)

        self.assertIs(
            detector_loader.call_args.kwargs["weights"],
            SSD300_VGG16_Weights.COCO_V1,
        )
        self.assertEqual(len(model.features), 30)
        self.assertTrue(model.features[16].ceil_mode)
        self.assertIsInstance(model.features[21], torch.nn.Conv2d)
        self.assertIsInstance(model.features[28], torch.nn.Conv2d)
        self.assertTrue(model.training is False)
        self.assertEqual(len(get_canonizer("vgg16_ssd_coco")), 1)

    def test_feature_resolutions_match_ssd300(self):
        with patch("models.vgg.ssd300_vgg16", return_value=self.detector):
            model = get_vgg16_ssd_coco(pretrained=False)
        x = torch.zeros(1, 3, 300, 300)
        activation = x
        shapes = {}
        with torch.no_grad():
            for index, layer in enumerate(model.features):
                activation = layer(activation)
                if index in {21, 28}:
                    shapes[index] = tuple(activation.shape)
            output = model(x)
        self.assertEqual(shapes[21], (1, 512, 38, 38))
        self.assertEqual(shapes[28], (1, 512, 19, 19))
        self.assertEqual(output.shape, (1, 512))

    def test_existing_activation_extraction_accepts_ssd_vgg(self):
        with patch("models.vgg.ssd300_vgg16", return_value=self.detector):
            model = get_vgg16_ssd_coco(pretrained=False)
        features = _get_features(
            torch.zeros(1, 3, 300, 300),
            "features.28",
            CondAttribution(model),
            EpsilonPlusFlat(
                canonizers=get_canonizer("vgg16_ssd_coco")
            ),
            "max",
            "cpu",
        )
        self.assertEqual(features.shape, (1, 512))

    def test_existing_lrp_localization_accepts_ssd_vgg(self):
        with patch("models.vgg.ssd300_vgg16", return_value=self.detector):
            model = get_vgg16_ssd_coco(pretrained=False)
        heatmap = get_localization(
            torch.randn(1, 512),
            torch.randn(1, 3, 64, 64),
            model,
            get_canonizer("vgg16_ssd_coco"),
            "features.28",
            "max",
            "cpu",
            model_name="vgg16_ssd_coco",
        )
        self.assertEqual(heatmap.shape, (1, 64, 64))
        self.assertTrue(torch.isfinite(heatmap).all())


if __name__ == "__main__":
    unittest.main()
