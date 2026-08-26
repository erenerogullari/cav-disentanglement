import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from datasets.coco.coco import CocoConceptDataset


class _MiniCOCO:
    def __init__(self, annotation_path):
        with open(annotation_path) as file:
            payload = json.load(file)
        self.images = {int(item["id"]): item for item in payload["images"]}
        self.categories = {int(item["id"]): item for item in payload["categories"]}
        self.annotations = {int(item["id"]): item for item in payload["annotations"]}

    def getCatIds(self):
        return list(self.categories)

    def loadCats(self, ids):
        return [self.categories[int(item)] for item in ids]

    def getImgIds(self, catIds=None):
        if not catIds:
            return list(self.images)
        required = set(int(item) for item in catIds)
        return [
            image_id
            for image_id in self.images
            if required.issubset(
                {
                    int(annotation["category_id"])
                    for annotation in self.annotations.values()
                    if int(annotation["image_id"]) == image_id
                }
            )
        ]

    def loadImgs(self, ids):
        return [self.images[int(item)] for item in ids]

    def getAnnIds(self, imgIds=None, catIds=None):
        image_ids = set(int(item) for item in (imgIds or []))
        category_ids = set(int(item) for item in (catIds or []))
        return [
            annotation_id
            for annotation_id, annotation in self.annotations.items()
            if (not image_ids or int(annotation["image_id"]) in image_ids)
            and (not category_ids or int(annotation["category_id"]) in category_ids)
        ]

    def loadAnns(self, ids):
        return [self.annotations[int(item)] for item in ids]

    def annToMask(self, annotation):
        return np.asarray(annotation["mask"], dtype=np.uint8)


class CocoConceptDatasetTest(unittest.TestCase):
    def _make_root(self, root: Path, n_images: int = 8) -> Path:
        (root / "train2017").mkdir(parents=True)
        (root / "val2017").mkdir(parents=True)
        (root / "annotations").mkdir(parents=True)
        categories = [
            {"id": index + 1, "name": name}
            for index, name in enumerate(
                ["cup", "fork", "knife", "spoon", "bowl", "dining table"]
            )
        ]
        images = []
        annotations = []
        annotation_id = 1
        for image_id in range(1, n_images + 1):
            filename = f"{image_id:012d}.jpg"
            images.append(
                {"id": image_id, "file_name": filename, "height": 4, "width": 4}
            )
            image = np.full((4, 4, 3), image_id, dtype=np.uint8)
            for split in ("train2017", "val2017"):
                Image.fromarray(image).save(root / split / filename)

            cup_mask = np.zeros((4, 4), dtype=np.uint8)
            cup_mask[0, 0] = 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "mask": cup_mask.tolist(),
                }
            )
            annotation_id += 1
            if image_id == 1:
                second_cup = np.zeros((4, 4), dtype=np.uint8)
                second_cup[3, 3] = 1
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "mask": second_cup.tolist(),
                    }
                )
                annotation_id += 1
                fork_mask = np.zeros((4, 4), dtype=np.uint8)
                fork_mask[1:3, 1:3] = 1
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 2,
                        "mask": fork_mask.tolist(),
                    }
                )
                annotation_id += 1

        payload = {
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }
        for split in ("train2017", "val2017"):
            with open(root / "annotations" / f"instances_{split}.json", "w") as file:
                json.dump(payload, file)
        return root

    def test_labels_category_resolution_and_union_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_root(Path(tmpdir))
            with patch("datasets.coco.coco.COCO", _MiniCOCO):
                dataset = CocoConceptDataset(root, image_size=8, max_samples=None)

            self.assertEqual(
                dataset.get_concept_names(),
                ["cup", "fork", "knife", "spoon", "bowl", "dining table"],
            )
            self.assertEqual(dataset.get_labels().shape, (8, 6))
            self.assertEqual(dataset.get_labels()[0].tolist(), [1, 1, 0, 0, 0, 0])
            masks = dataset.get_concept_masks(0)
            self.assertTrue(masks["cup"][0, 0])
            self.assertTrue(masks["cup"][-1, -1])
            self.assertEqual(masks["cup"].shape, (8, 8))
            self.assertEqual(dataset[0][0].shape, (3, 8, 8))

    def test_uniform_subset_is_deterministic_and_in_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_root(Path(tmpdir))
            with patch("datasets.coco.coco.COCO", _MiniCOCO):
                first = CocoConceptDataset(root, max_samples=4, subset_seed=42)
                second = CocoConceptDataset(root, max_samples=4, subset_seed=42)

            np.testing.assert_array_equal(first.image_ids, second.image_ids)
            expected = np.sort(
                np.random.default_rng(42).choice(np.arange(1, 9), size=4, replace=False)
            )
            np.testing.assert_array_equal(first.image_ids, expected)
            self.assertIn("train2017_dining6_300_n4_seed42", first.cache_identity)

    def test_unknown_concept_and_missing_layout_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_root(Path(tmpdir))
            with patch("datasets.coco.coco.COCO", _MiniCOCO):
                with self.assertRaisesRegex(ValueError, "Unknown COCO concept"):
                    CocoConceptDataset(root, concepts=["not-a-category"])
            with patch("datasets.coco.coco.COCO", _MiniCOCO):
                with self.assertRaisesRegex(FileNotFoundError, "image directory"):
                    CocoConceptDataset(Path(tmpdir) / "missing")


if __name__ == "__main__":
    unittest.main()
