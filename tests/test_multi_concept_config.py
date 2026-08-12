import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir


class MultiConceptConfigTest(unittest.TestCase):
    def test_config_composes_for_all_k_and_cav_targets(self):
        config_dir = str(Path(__file__).parents[1] / "experiment_configs")
        targets = {
            "pattern_cav": "cav_models.PatternCAV",
            "multi_cav": "cav_models.MultiPatternCAV",
            "log_cav": "cav_models.LogCAV",
            "svm_cav": "cav_models.SvmCAV",
            "ridge_cav": "cav_models.RidgeCAV",
            "random_cav": "cav_models.RandomCAV",
        }
        with patch.dict(
            os.environ,
            {"CELEBA_DATA_ROOT": "/tmp/celeba", "CLEAN_VGG16_CHECKPOINT": "/tmp/clean.pth"},
        ):
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                for k in (1, 3, 5):
                    for cav_name, target in targets.items():
                        cfg = compose(
                            config_name="multi_concept_alignment",
                            overrides=[
                                f"dataset.num_concepts={k}",
                                f"cav._target_={target}",
                                f"cav.name={cav_name}",
                            ],
                        )
                        self.assertEqual(cfg.dataset.num_concepts, k)
                        self.assertEqual(cfg.cav._target_, target)
                        self.assertEqual(cfg.cav.name, cav_name)
                        self.assertEqual(cfg.cav.layer, "features.29")
                        self.assertEqual(cfg.train.num_epochs, 50)
                        self.assertFalse(cfg.localization.concept_cleaning.enabled)


if __name__ == "__main__":
    unittest.main()
