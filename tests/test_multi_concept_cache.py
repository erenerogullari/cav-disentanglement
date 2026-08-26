import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from experiments.multi_concept_alignment.variant import (
    build_experiment_signature,
    build_variant_id,
    ensure_classifier_checkpoint,
    validate_classifier_checkpoint,
)


def _config(num_concepts=3):
    return OmegaConf.create(
        {
            "dataset": {
                "_target_": "datasets.get_celeba_attacked_dataset",
                "name": "celeba_attacked",
                "data_paths": ["/tmp/celeba"],
                "normalize_data": True,
                "attacked_classes": [1],
                "num_concepts": num_concepts,
                "p_artifact": 0.4,
                "cooccurrence_probability": 0.5,
                "entanglement_factor": 5,
                "artifact_seed": 0,
                "seed": 42,
                "val_split": 0.1,
                "test_split": 0.1,
                "image_size": 224,
                "artifact_type": "ch_time",
                "time_format": "datetime",
                "brightness_factor": 1.4,
                "cache_namespace": "pending",
            },
            "model": {
                "name": "vgg16",
                "clean_ckpt_path": "/tmp/clean.pth",
                "ckpt_path": None,
                "pretrained": False,
                "n_class": 2,
            },
            "classifier_train": {
                "device": "cpu",
                "num_workers": 0,
                "batch_size": 2,
                "random_seed": 42,
                "num_epochs": 20,
                "learning_rate": 1e-4,
                "val_split": 0.1,
                "test_split": 0.1,
                "log_interval": 1,
                "save_best": True,
            },
        }
    )


class MultiConceptCacheTest(unittest.TestCase):
    def test_k_values_have_distinct_variant_namespaces(self):
        signatures = [build_experiment_signature(_config(k)) for k in range(1, 6)]
        variants = [build_variant_id(signature) for signature in signatures]
        self.assertEqual(len(set(variants)), 5)
        self.assertTrue(all(variant.startswith(f"k{k}-") for k, variant in enumerate(variants, 1)))

    def test_matching_checkpoint_skips_training(self):
        cfg = _config()
        signature = build_experiment_signature(cfg)
        variant = build_variant_id(signature)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = (
                Path(temp_dir)
                / "checkpoints"
                / "celeba_attacked"
                / variant
                / "checkpoint_vgg16.pth"
            )
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {"model_state_dict": {"weight": torch.ones(1)}, "experiment_signature": signature},
                checkpoint,
            )
            calls = []

            def fail_if_called(*args, **kwargs):
                calls.append((args, kwargs))
                raise AssertionError("training should have been skipped")

            actual_path, trained = ensure_classifier_checkpoint(
                cfg, Path(temp_dir), signature, variant, train_fn=fail_if_called
            )
            self.assertEqual(actual_path, checkpoint)
            self.assertFalse(trained)
            self.assertEqual(calls, [])

    def test_mismatched_and_malformed_checkpoints_are_rejected(self):
        signature = build_experiment_signature(_config())
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pth"
            torch.save(
                {"model_state_dict": {}, "experiment_signature": {"wrong": True}},
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_classifier_checkpoint(checkpoint, signature)

            checkpoint.write_bytes(b"not a torch checkpoint")
            with self.assertRaisesRegex(ValueError, "Could not load"):
                validate_classifier_checkpoint(checkpoint, signature)


if __name__ == "__main__":
    unittest.main()
