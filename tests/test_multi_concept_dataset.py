import unittest

import numpy as np
import pandas as pd
import torch
from PIL import Image

from datasets.celeba.celeba_attacked import (
    ARTIFACT_CONCEPT_POOL,
    CelebAAttackedDataset,
    generate_artifact_labels,
    get_active_artifact_concepts,
)
from datasets.celeba.artificial_artifact import insert_artifact


class MultiConceptLabelTest(unittest.TestCase):
    def test_num_concepts_bounds_and_prefix_order(self):
        self.assertEqual(get_active_artifact_concepts(1), ["timestamp"])
        self.assertEqual(get_active_artifact_concepts(3), ARTIFACT_CONCEPT_POOL[:3])
        self.assertEqual(get_active_artifact_concepts(5), ARTIFACT_CONCEPT_POOL)
        for invalid in (0, 6):
            with self.assertRaises(ValueError):
                get_active_artifact_concepts(invalid)

    def test_k3_matches_original_label_generation_order(self):
        targets = np.array([0, 1, 0, 1, 1, 0] * 20)
        actual = generate_artifact_labels(
            targets,
            attacked_classes=[1],
            num_concepts=3,
            p_artifact=0.4,
            cooccurrence_probability=0.5,
            entanglement_factor=5,
            artifact_seed=0,
        )

        rng = np.random.RandomState(0)
        timestamp = rng.rand(len(targets)) < np.where(targets == 1, 0.4, 0.005)
        box = rng.rand(len(targets)) < np.where(timestamp, 0.5, 0.1)
        brightness = rng.rand(len(targets)) < np.where(timestamp, 0.5, 0.1)
        np.testing.assert_array_equal(actual["timestamp"], timestamp)
        np.testing.assert_array_equal(actual["box"], box)
        np.testing.assert_array_equal(actual["brightness"], brightness)
        self.assertEqual(list(actual), ARTIFACT_CONCEPT_POOL[:3])

    def test_shared_label_vectors_are_identical_across_k(self):
        targets = np.arange(1000) % 2
        labels_by_k = {
            k: generate_artifact_labels(targets, [1], num_concepts=k)
            for k in range(1, 6)
        }
        for smaller_k in range(1, 5):
            for concept in ARTIFACT_CONCEPT_POOL[:smaller_k]:
                np.testing.assert_array_equal(
                    labels_by_k[smaller_k][concept], labels_by_k[5][concept]
                )

    def test_additional_concept_conditional_frequencies(self):
        targets = np.ones(200_000, dtype=int)
        labels = generate_artifact_labels(
            targets,
            attacked_classes=[1],
            num_concepts=5,
            p_artifact=0.4,
            cooccurrence_probability=0.5,
            entanglement_factor=5,
            artifact_seed=7,
        )
        timestamp = labels["timestamp"]
        for concept in ARTIFACT_CONCEPT_POOL[1:]:
            self.assertAlmostEqual(labels[concept][timestamp].mean(), 0.5, delta=0.01)
            self.assertAlmostEqual(
                labels[concept][~timestamp].mean(), 0.1, delta=0.01
            )


class MultiConceptArtifactTest(unittest.TestCase):
    def test_subset_labels_and_concept_names_follow_active_prefix(self):
        for k in (1, 3, 5):
            dataset = CelebAAttackedDataset.__new__(CelebAAttackedDataset)
            dataset.artifact_concepts = ARTIFACT_CONCEPT_POOL[:k]
            dataset.attributes = pd.DataFrame(
                {"Smiling": [0, 1, 0], "Young": [1, 0, 1]}
            )
            dataset.metadata = dataset.attributes.copy()
            dataset.metadata["targets"] = [0, 1, 0]
            dataset.artifact_labels_by_concept = {
                concept: np.array([True, False, True])
                for concept in dataset.artifact_concepts
            }
            for concept, labels in dataset.artifact_labels_by_concept.items():
                dataset.metadata[concept] = labels.astype(int)
            dataset.sample_ids_by_concept = {
                concept: np.where(dataset.attributes[concept].to_numpy() == 1)[0]
                for concept in dataset.attributes.columns
            }

            subset = dataset.get_subset_by_idxs([0, 2])
            self.assertEqual(tuple(subset.get_labels().shape), (2, 2 + k))
            self.assertEqual(
                subset.get_concept_names(),
                ["Smiling", "Young", *ARTIFACT_CONCEPT_POOL[:k]],
            )
            self.assertEqual(
                list(subset.sample_ids_by_artifact), ARTIFACT_CONCEPT_POOL[:k]
            )

    def test_checkerboard_and_watermark_are_deterministic_and_localized(self):
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        for artifact_type in ("checkerboard", "watermark"):
            np.random.seed(4)
            image_a, mask_a = insert_artifact(image.copy(), artifact_type)
            np.random.seed(4)
            image_b, mask_b = insert_artifact(image.copy(), artifact_type)
            np.testing.assert_array_equal(np.asarray(image_a), np.asarray(image_b))
            torch.testing.assert_close(mask_a, mask_b)
            self.assertEqual(tuple(mask_a.shape), (224, 224))
            self.assertGreater(int(mask_a.sum()), 0)
            self.assertLess(int(mask_a.sum()), 32 * 32)

        np.random.seed(4)
        _, watermark_mask = insert_artifact(image.copy(), "watermark")
        ys, xs = torch.where(watermark_mask.bool())
        bounding_box_area = int(
            (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        )
        self.assertLess(int(watermark_mask.sum()), bounding_box_area)

    def test_rendered_masks_expose_only_active_prefix_and_do_not_overlap(self):
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        all_specs = {
            "timestamp": ("ch_time", {"img_size": 224}),
            "box": ("random_box", {}),
            "brightness": ("brightness", {"factor": 1.4}),
            "checkerboard": ("checkerboard", {}),
            "watermark": ("watermark", {}),
        }
        masks_by_k = {}
        for k in (1, 3, 5):
            dataset = CelebAAttackedDataset.__new__(CelebAAttackedDataset)
            dataset.image_size = 224
            dataset.artifact_seed = 0
            dataset.artifact_concepts = ARTIFACT_CONCEPT_POOL[:k]
            dataset.artifact_specs = {
                concept: all_specs[concept] for concept in dataset.artifact_concepts
            }
            dataset.artifact_labels_by_concept = {
                concept: np.array([True]) for concept in dataset.artifact_concepts
            }
            _, masks = dataset.render_artifacts(image.copy(), 0)
            masks_by_k[k] = masks
            self.assertEqual(list(masks), ARTIFACT_CONCEPT_POOL[:k])
            localized = [
                masks[concept].bool()
                for concept in dataset.artifact_concepts
                if concept != "brightness"
            ]
            for left_index, left in enumerate(localized):
                for right in localized[left_index + 1 :]:
                    self.assertFalse(torch.logical_and(left, right).any())
        for concept in ARTIFACT_CONCEPT_POOL[:3]:
            torch.testing.assert_close(masks_by_k[3][concept], masks_by_k[5][concept])


if __name__ == "__main__":
    unittest.main()
