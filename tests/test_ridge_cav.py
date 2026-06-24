import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from cav_models.ridge_cav import RidgeCAV
from experiments.utils.cav_model_utils import instantiate_cav_model
from utils.cav import compute_cavs


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    weights = (y == 1) / n_positive + (y == 0) / n_negative
    return weights / weights.max()


class RidgeCAVTest(unittest.TestCase):
    def test_config_instantiates_ridge_cav(self):
        config_path = (
            Path(__file__).parents[1] / "configs" / "cav_model" / "ridge_cav.yaml"
        )
        model = instantiate_cav_model(
            OmegaConf.load(config_path),
            n_concepts=2,
            n_features=3,
        )

        self.assertIsInstance(model, RidgeCAV)

    def test_compute_cavs_matches_independent_ridge_grid_searches(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(160, 8))
        true_weights = rng.normal(size=(3, 8))
        logits = x @ true_weights.T + 0.4 * rng.normal(size=(160, 3))
        labels = (logits > np.array([0.0, 0.5, -0.5])).astype(np.float64)
        labels = np.column_stack([labels, np.zeros(labels.shape[0])])

        expected_weights = []
        expected_bias = []
        for concept_idx in range(3):
            y = labels[:, concept_idx]
            grid_search = GridSearchCV(
                Ridge(fit_intercept=True),
                param_grid={"alpha": [10**i for i in range(-5, 5)]},
            )
            grid_search.fit(
                x,
                y * 2.0 - 1.0,
                sample_weight=_balanced_weights(y),
            )
            expected_weights.append(grid_search.best_estimator_.coef_)
            expected_bias.append(grid_search.best_estimator_.intercept_)
        expected_weights.append(np.zeros(x.shape[1]))
        expected_bias.append(0.0)

        weights, bias = compute_cavs(
            torch.from_numpy(x),
            torch.from_numpy(labels),
            type="ridge_cav",
            normalize=False,
        )

        np.testing.assert_allclose(weights.numpy(), np.stack(expected_weights))
        np.testing.assert_allclose(bias.numpy(), np.array(expected_bias))

    def test_train_step_uses_signed_target_mse(self):
        model = RidgeCAV(n_concepts=2, n_features=3)
        weights = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
        bias = torch.tensor([0.25, -0.25])
        model.set_params(weights, bias)
        x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        cav_loss, orthogonality_loss = model.train_step(x, y, torch.zeros(2, 2))
        expected_loss = torch.nn.functional.mse_loss(model(x), y * 2.0 - 1.0)

        torch.testing.assert_close(cav_loss, expected_loss)
        torch.testing.assert_close(orthogonality_loss, torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
