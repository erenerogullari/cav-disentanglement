import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from datasets.base_dataset import BaseDataset

try:
    from pycocotools.coco import COCO
except ModuleNotFoundError:  # pragma: no cover - exercised through the error path
    COCO = None  # type: ignore[assignment]


COCO_CONCEPT_PRESETS = {
    "dining6": ("cup", "fork", "knife", "spoon", "bowl", "dining table"),
}

SSD_IMAGE_MEAN = (0.48235, 0.45882, 0.40784)
SSD_IMAGE_STD = (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)


def _resolve_root(data_paths: Sequence[str] | str | Path) -> Path:
    if isinstance(data_paths, (str, Path)):
        return Path(data_paths).expanduser()
    if len(data_paths) != 1:
        raise ValueError(
            f"COCO expects exactly one dataset root, got {len(data_paths)} paths."
        )
    return Path(data_paths[0]).expanduser()


def _resolve_concepts(
    concept_set: str | None, concepts: Sequence[str] | None
) -> tuple[list[str], str]:
    if concepts is not None:
        resolved = [str(concept) for concept in concepts]
        if not resolved:
            raise ValueError("Custom COCO concept lists must not be empty.")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"COCO concept names must be unique, got {resolved}.")
        return resolved, "custom"

    preset = "dining6" if concept_set is None else str(concept_set)
    if preset not in COCO_CONCEPT_PRESETS:
        raise ValueError(
            f"Unknown COCO concept preset '{preset}'. "
            f"Available presets: {sorted(COCO_CONCEPT_PRESETS)}."
        )
    return list(COCO_CONCEPT_PRESETS[preset]), preset


def get_coco_dataset(
    data_paths: Sequence[str] | str,
    normalize_data: bool = True,
    image_size: int = 300,
    split: str = "train2017",
    concept_set: str = "dining6",
    concepts: Sequence[str] | None = None,
    max_samples: int | None = 10000,
    subset_seed: int = 42,
    **kwargs,
) -> "CocoConceptDataset":
    return CocoConceptDataset(
        data_paths=data_paths,
        normalize_data=normalize_data,
        image_size=image_size,
        split=split,
        concept_set=concept_set,
        concepts=concepts,
        max_samples=max_samples,
        subset_seed=subset_seed,
        **kwargs,
    )


class CocoConceptDataset(BaseDataset):
    """COCO image-presence concepts with instance-union localization masks."""

    def __init__(
        self,
        data_paths: Sequence[str] | str,
        normalize_data: bool = True,
        image_size: int = 300,
        split: str = "train2017",
        concept_set: str = "dining6",
        concepts: Sequence[str] | None = None,
        max_samples: int | None = 10000,
        subset_seed: int = 42,
        **kwargs,
    ) -> None:
        if COCO is None:
            raise ModuleNotFoundError(
                "COCO support requires pycocotools. Install requirements.txt first."
            )
        if kwargs:
            # Dataset configs in this repository commonly carry generic keys such as
            # ``shuffle``. They are intentionally ignored by dataset adapters.
            kwargs.clear()

        root = _resolve_root(data_paths)
        image_dir = root / split
        annotation_path = root / "annotations" / f"instances_{split}.json"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {image_dir}")
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"COCO instance annotations not found: {annotation_path}"
            )

        self.root = root
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.split = str(split)
        self.image_size = int(image_size)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.subset_seed = int(subset_seed)
        self.concept_names, self.concept_set = _resolve_concepts(
            concept_set, concepts
        )

        transform_steps: list = [
            T.Resize(
                (self.image_size, self.image_size),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
        ]
        if normalize_data:
            transform_steps.append(T.Normalize(SSD_IMAGE_MEAN, SSD_IMAGE_STD))
        transform = T.Compose(transform_steps)
        super().__init__([str(root)], transform=transform)

        self.mean = torch.tensor(
            SSD_IMAGE_MEAN if normalize_data else (0.0, 0.0, 0.0)
        )
        self.var = torch.tensor(
            SSD_IMAGE_STD if normalize_data else (1.0, 1.0, 1.0)
        )
        self.normalize_fn = (
            T.Normalize(SSD_IMAGE_MEAN, SSD_IMAGE_STD)
            if normalize_data
            else torch.nn.Identity()
        )

        self.coco = COCO(str(annotation_path))
        categories = self.coco.loadCats(self.coco.getCatIds())
        category_id_by_name = {str(cat["name"]): int(cat["id"]) for cat in categories}
        missing = [name for name in self.concept_names if name not in category_id_by_name]
        if missing:
            raise ValueError(
                f"Unknown COCO concept names: {missing}. "
                f"Available names include: {sorted(category_id_by_name)[:10]}..."
            )
        self.category_ids = [category_id_by_name[name] for name in self.concept_names]
        self.category_id_by_name = dict(zip(self.concept_names, self.category_ids))

        image_ids = np.array(sorted(self.coco.getImgIds()), dtype=np.int64)
        if self.max_samples is not None:
            if self.max_samples <= 0:
                raise ValueError(
                    f"max_samples must be positive or null, got {self.max_samples}."
                )
            if self.max_samples < len(image_ids):
                rng = np.random.default_rng(self.subset_seed)
                image_ids = np.sort(
                    rng.choice(image_ids, size=self.max_samples, replace=False)
                )
        self.image_ids = image_ids
        self._refresh_labels_and_indices()

        concept_key = (
            self.concept_set
            if self.concept_set != "custom"
            else "custom-" + "-".join(name.replace(" ", "-") for name in self.concept_names)
        )
        sample_key = "all" if self.max_samples is None else str(self.max_samples)
        self.cache_identity = (
            f"coco_{self.split}_{concept_key}_{self.image_size}_"
            f"n{sample_key}_seed{self.subset_seed}"
        )

        self.classes = list(self.concept_names)
        self.class_names = list(self.concept_names)
        self.num_classes = len(self.concept_names)
        counts = self.labels.sum(dim=0).numpy()
        self.weights = self.compute_weights(counts)

    def _refresh_labels_and_indices(self) -> None:
        image_id_to_index = {
            int(image_id): index for index, image_id in enumerate(self.image_ids)
        }
        labels = torch.zeros(
            (len(self.image_ids), len(self.concept_names)), dtype=torch.float32
        )
        for concept_index, category_id in enumerate(self.category_ids):
            positive_image_ids = self.coco.getImgIds(catIds=[category_id])
            positive_indices = [
                image_id_to_index[int(image_id)]
                for image_id in positive_image_ids
                if int(image_id) in image_id_to_index
            ]
            if positive_indices:
                labels[positive_indices, concept_index] = 1.0
        self.labels = labels
        self.sample_ids_by_concept = {
            name: torch.nonzero(labels[:, index] > 0, as_tuple=False)
            .squeeze(1)
            .numpy()
            for index, name in enumerate(self.concept_names)
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_info = self.coco.loadImgs([int(self.image_ids[index])])[0]
        image_path = self.image_dir / str(image_info["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {image_path}")
        with Image.open(image_path) as image_raw:
            image = image_raw.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)
        return image.float(), self.labels[index].clone()

    def get_concept_masks(self, index: int) -> dict[str, torch.Tensor]:
        image_id = int(self.image_ids[index])
        image_info = self.coco.loadImgs([image_id])[0]
        height, width = int(image_info["height"]), int(image_info["width"])
        masks: dict[str, torch.Tensor] = {}
        for concept_name, category_id in zip(self.concept_names, self.category_ids):
            union = np.zeros((height, width), dtype=np.uint8)
            annotation_ids = self.coco.getAnnIds(
                imgIds=[image_id], catIds=[category_id]
            )
            for annotation in self.coco.loadAnns(annotation_ids):
                union |= self.coco.annToMask(annotation).astype(np.uint8)
            mask = torch.from_numpy(union).unsqueeze(0).float()
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=T.InterpolationMode.NEAREST,
                antialias=False,
            ).squeeze(0)
            masks[concept_name] = mask.bool()
        return masks

    def get_all_ids(self) -> list[int]:
        return [int(image_id) for image_id in self.image_ids]

    def get_sample_name(self, index: int) -> str:
        return str(int(self.image_ids[index]))

    def get_target(self, index: int) -> torch.Tensor:
        return self.labels[index].clone()

    def get_labels(self) -> torch.Tensor:
        return self.labels.clone()

    def get_concept_names(self) -> list[str]:
        return list(self.concept_names)

    def get_class_names(self) -> list[str]:
        return list(self.class_names)

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_subset_by_idxs(self, idxs: Sequence[int]) -> "CocoConceptDataset":
        subset = copy.copy(self)
        subset.image_ids = self.image_ids[list(idxs)].copy()
        subset._refresh_labels_and_indices()
        return subset
