import unittest

import torch
import torch.nn.functional as F

from cav_models.multi_cav import MultiPatternCAV
from cav_models.pattern_cav import PatternCAV


class MultiPatternCAVTest(unittest.TestCase):
    def test_train_and_val_losses_scale_dynamically_with_concept_count(self):
        n_features = 4
        batch_size = 3

        for n_concepts in (1, 2, 5):
            with self.subTest(n_concepts=n_concepts):
                model = MultiPatternCAV(n_concepts, n_features)
                weights = (
                    torch.arange(1, n_concepts * n_features + 1, dtype=torch.float32)
                    .reshape(n_concepts, n_features)
                    / 10.0
                )
                bias = torch.linspace(-0.2, 0.2, n_features).unsqueeze(0)
                model.set_params(weights, bias)

                labels = (
                    torch.arange(batch_size * n_concepts)
                    .reshape(batch_size, n_concepts)
                    .remainder(2)
                    .float()
                )
                x = torch.linspace(
                    -1.0, 1.0, batch_size * n_features
                ).reshape(batch_size, n_features)
                pair_weights = (
                    torch.arange(1, n_concepts**2 + 1, dtype=torch.float32)
                    .reshape(n_concepts, n_concepts)
                    / 7.0
                )

                raw_mse = F.mse_loss(model(labels), x)
                gram = weights @ weights.T
                expected_cav_loss = raw_mse / n_concepts
                expected_orthogonality_loss = torch.norm(
                    pair_weights * (gram - torch.eye(n_concepts)),
                    p="fro",
                )

                train_losses = model.train_step(x, labels, pair_weights)
                val_losses = model.val_step(x, labels, pair_weights)

                torch.testing.assert_close(train_losses[0], expected_cav_loss)
                torch.testing.assert_close(
                    train_losses[1], expected_orthogonality_loss
                )
                torch.testing.assert_close(val_losses[0], expected_cav_loss)
                torch.testing.assert_close(
                    val_losses[1], expected_orthogonality_loss
                )

    def test_orthogonality_value_and_gradient_match_pattern_cav(self):
        n_concepts = 4
        n_features = 3
        weights = torch.tensor(
            [
                [0.8, -0.1, 0.2],
                [0.3, 0.7, -0.4],
                [-0.2, 0.5, 0.6],
                [0.1, -0.3, 0.9],
            ]
        )
        pair_weights = torch.tensor(
            [
                [1.0, 0.5, 0.7, 0.2],
                [0.5, 1.0, 0.4, 0.8],
                [0.7, 0.4, 1.0, 0.6],
                [0.2, 0.8, 0.6, 1.0],
            ]
        )
        x = torch.zeros(2, n_features)
        labels = torch.zeros(2, n_concepts)

        pattern = PatternCAV(n_concepts, n_features)
        pattern.set_params(weights.clone(), torch.zeros(n_concepts, n_features))
        multi = MultiPatternCAV(n_concepts, n_features)
        multi.set_params(weights.clone(), torch.zeros(1, n_features))

        _, pattern_orthogonality = pattern.train_step(x, labels, pair_weights)
        _, multi_orthogonality = multi.train_step(x, labels, pair_weights)
        pattern_gradient = torch.autograd.grad(
            pattern_orthogonality, pattern.weights
        )[0]
        multi_gradient = torch.autograd.grad(
            multi_orthogonality, multi.weights
        )[0]

        torch.testing.assert_close(
            multi_orthogonality, pattern_orthogonality
        )
        torch.testing.assert_close(multi_gradient, pattern_gradient)

    def test_scaled_reconstruction_gradient_matches_pattern_for_equal_residuals(self):
        n_concepts = 4
        n_features = 3
        x = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]])
        labels = torch.ones(x.shape[0], n_concepts)
        pair_weights = torch.zeros(n_concepts, n_concepts)

        pattern = PatternCAV(n_concepts, n_features)
        pattern.set_params(
            torch.zeros(n_concepts, n_features),
            torch.zeros(n_concepts, n_features),
        )
        multi = MultiPatternCAV(n_concepts, n_features)
        multi.set_params(
            torch.zeros(n_concepts, n_features),
            torch.zeros(1, n_features),
        )

        pattern_cav_loss, _ = pattern.train_step(x, labels, pair_weights)
        multi_cav_loss, _ = multi.train_step(x, labels, pair_weights)
        pattern_gradient = torch.autograd.grad(pattern_cav_loss, pattern.weights)[0]
        multi_gradient = torch.autograd.grad(multi_cav_loss, multi.weights)[0]

        torch.testing.assert_close(multi_gradient, pattern_gradient)


if __name__ == "__main__":
    unittest.main()
