import unittest

import torch

from cav_models.g_sae import G_SAE
from cav_models.log_cav import LogCAV
from cav_models.random_cav import RandomCAV
from cav_models.ridge_cav import RidgeCAV
from cav_models.svm_cav import SvmCAV


class OrthogonalityScalingTest(unittest.TestCase):
    def setUp(self):
        self.n_concepts = 2
        self.n_features = 3
        self.weights = torch.tensor(
            [[0.8, -0.1, 0.2], [0.3, 0.7, -0.4]]
        )
        self.pair_weights = torch.tensor([[1.0, 0.5], [0.5, 0.8]])
        self.x = torch.tensor(
            [[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0], [0.3, 0.4, -0.7]]
        )
        self.labels = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )

    def _expected_orthogonality_loss(self, directions):
        gram = directions @ directions.T
        return torch.norm(
            self.pair_weights * (gram - torch.eye(self.n_concepts)),
            p="fro",
        )

    def test_linear_cav_models_use_unnormalized_frobenius_loss(self):
        model_classes = (LogCAV, SvmCAV, RidgeCAV, RandomCAV)

        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                model = model_class(self.n_concepts, self.n_features)
                model.set_params(
                    self.weights.clone(), torch.zeros(self.n_concepts)
                )
                expected = self._expected_orthogonality_loss(self.weights)

                _, train_orthogonality = model.train_step(
                    self.x, self.labels, self.pair_weights
                )
                _, val_orthogonality = model.val_step(
                    self.x, self.labels, self.pair_weights
                )

                torch.testing.assert_close(train_orthogonality, expected)
                torch.testing.assert_close(val_orthogonality, expected)

    def test_g_sae_uses_unnormalized_frobenius_loss(self):
        model = G_SAE(
            self.n_concepts,
            self.n_features,
            n_latents=self.n_features,
        )
        with torch.no_grad():
            model.decoder.weight[:, : self.n_concepts].copy_(self.weights.T)

        expected = self._expected_orthogonality_loss(self.weights)
        _, train_orthogonality = model.train_step(
            self.x, self.labels, self.pair_weights
        )
        _, val_orthogonality = model.val_step(
            self.x, self.labels, self.pair_weights
        )

        torch.testing.assert_close(train_orthogonality, expected)
        torch.testing.assert_close(val_orthogonality, expected)


if __name__ == "__main__":
    unittest.main()
