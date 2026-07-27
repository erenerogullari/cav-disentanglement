import types
import unittest

import torch
from torchvision.models.vision_transformer import VisionTransformer

from utils.reslrp_torchvision import attribute, attribute_concept, self_check


def _tiny_vit() -> VisionTransformer:
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=3,
    )
    model.inspection_layer = torch.nn.Identity()

    def forward(self, x):
        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = self.encoder(torch.cat([class_token, x], dim=1))[:, 0]
        return self.heads(self.inspection_layer(x))

    model.forward = types.MethodType(forward, model)
    return model


class ResLRPTorchvisionTest(unittest.TestCase):
    def test_self_check_and_class_attribution(self):
        model = _tiny_vit().eval()

        result = self_check(model, verbose=False)
        heatmap = attribute(model, torch.randn(2, 3, 32, 32))

        self.assertEqual(result["depth"], 2)
        self.assertEqual(result["counts"]["ResidualGamma"], 4)
        self.assertEqual(result["counts"]["GammaLinear"], 4)
        self.assertEqual(result["counts"]["ZPlusConv2d"], 1)
        self.assertEqual(tuple(heatmap.shape), (2, 32, 32))
        self.assertTrue(torch.isfinite(heatmap).all())

    def test_concept_attribution_restores_model_state(self):
        model = _tiny_vit().train()
        first_parameter = next(model.parameters())
        first_parameter.requires_grad_(False)
        requires_grad_before = [parameter.requires_grad for parameter in model.parameters()]
        x = torch.randn(2, 3, 32, 32)
        cav = torch.randn(model.hidden_dim)

        logits_before = model.eval()(x).detach()
        model.train()
        heatmap = attribute_concept(model, x, cav, "inspection_layer")
        self.assertTrue(model.training)
        self.assertEqual(
            [parameter.requires_grad for parameter in model.parameters()],
            requires_grad_before,
        )
        logits_after = model.eval()(x).detach()

        self.assertEqual(tuple(heatmap.shape), (2, 32, 32))
        self.assertTrue(torch.isfinite(heatmap).all())
        torch.testing.assert_close(logits_before, logits_after)

    def test_concept_attribution_validates_layer_and_cav_shape(self):
        model = _tiny_vit().eval()
        x = torch.randn(1, 3, 32, 32)

        with self.assertRaisesRegex(ValueError, "unknown layer"):
            attribute_concept(model, x, torch.randn(model.hidden_dim), "missing")
        with self.assertRaisesRegex(ValueError, "CAV has"):
            attribute_concept(
                model,
                x,
                torch.randn(model.hidden_dim + 1),
                "inspection_layer",
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional CAV"):
            attribute_concept(
                model,
                x,
                torch.randn(2, model.hidden_dim),
                "inspection_layer",
            )


if __name__ == "__main__":
    unittest.main()
