import copy
import logging
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.base_dataset import BaseDataset
from utils.visualizations import visualize_sim_matrix
from .classes import ElementDataset
from .shapes import square, circle, triangle, cross, plus
from .colors import COLORS
from .textures import (
    solid,
    random_spots,
    regular_spots,
    polka,
    chequerboard,
    striped_diagonal,
    striped_vertical,
    striped_horizontal,
    striped_diagonal_alt,
)

log = logging.getLogger(__name__)


# Allowed shapes and textures
SHAPES = {
    "square": square,
    "circle": circle,
    "triangle": triangle,
    "cross": cross,
    "plus": plus,
}
TEXTURES = {
    "solid": {
        "plain": solid,
    },
    "spots": {
        "random": random_spots,
        "regular": regular_spots,
        "polka": polka,
        "chequerboard": chequerboard,
    },
    "stripes": {
        "horizontal": striped_horizontal,
        "vertical": striped_vertical,
        "diagonal": striped_diagonal,
        "diagonal_alt": striped_diagonal_alt,
    },
}

location_keywords = {
    "left": ["<127", ">0"],
    "right": [">127", ">0"],
    "bot": [">0", ">127"],
    "top": [">0", "<127"],
}


elements_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=0.3),
    T.RandomApply([T.RandomRotation(degrees=15)], p=0.2),
])

DEFAULT_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
DEFAULT_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_NUM_SAMPLES: int = 8192
DEFAULT_ELEMENT_COUNT: int = 3
DEFAULT_ELEMENT_SIZE: int = 96
DEFAULT_ELEMENT_SIZE_DELTA: int = 24
DEFAULT_ELEMENT_SEED: int = 2024
DEFAULT_LOC_SEED: int = 1337
DEFAULT_PLACE_REMAINING_RANDOMLY: bool = False

DEFAULT_ALLOWED_SHAPES: Tuple[str, ...] = tuple(sorted(SHAPES.keys()))
DEFAULT_ALLOWED_COLORS: Tuple[str, ...] = tuple(sorted(COLORS.keys()))
DEFAULT_ALLOWED_TEXTURES: Tuple[str, ...] = (
    "solid",
    *tuple(
        f"{family}_{variant}"
        for family, variants in TEXTURES.items()
        if family != "solid"
        for variant in sorted(variants.keys())
    ),
)



def _normalization_transform(normalize_data: bool, image_size: int) -> T.Compose:
    transforms: List[T.Compose] = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC), # type: ignore
        T.ToTensor(),
    ]
    if normalize_data:
        transforms.append(T.Normalize(DEFAULT_MEAN, DEFAULT_STD)) # type: ignore
    return T.Compose(transforms)


def _build_allowed_dict(
    shapes: Sequence[str],
    colors: Sequence[str],
    textures: Sequence[str],
    element_size: int,
    element_size_delta: int,
) -> Dict[str, Iterable[str] | Tuple[int, int]]:
    min_size = max(1, element_size - element_size_delta)
    max_size = max(min_size, element_size + element_size_delta)
    return {
        "sizes": (min_size, max_size),
        "shapes": list(shapes),
        "colors": list(colors),
        "textures": list(textures),
    }


def _default_class_configs(
    shapes: Sequence[str],
    colors: Sequence[str],
    textures: Sequence[str],
) -> Tuple[List[Dict[str, Optional[str]]], List[str]]:
    configs: List[Dict[str, Optional[str]]] = []
    names: List[str] = []

    for shape in shapes:
        configs.append({"shape": shape, "color": None, "texture": None})
        names.append(f"shape:{shape}")

    for color in colors:
        configs.append({"shape": None, "color": color, "texture": None})
        names.append(f"color:{color}")

    for texture in textures:
        configs.append({"shape": None, "color": None, "texture": texture})
        names.append(f"texture:{texture}")

    return configs, names


def _compose_class_name(idx: int, class_config: Dict[str, Optional[str]]) -> str:
    """
    Build a readable class name from the provided configuration.
    Any key with value None is skipped. If nothing remains, fall back to an index based name.
    """
    parts: List[str] = []
    for key in sorted(class_config.keys()):
        value = class_config[key]
        if value is None:
            continue
        parts.append(f"{key}:{value}")
    if not parts:
        return f"class:{idx}"
    return " ".join(parts)


def _compute_concept_counts(labels: torch.Tensor) -> np.ndarray:
    if labels.numel() == 0:
        return np.array([])
    return labels.sum(dim=0).detach().cpu().numpy()


def _pil_from_element_image(element_image) -> Image.Image:
    array = getattr(element_image, "img", element_image)
    array = np.asarray(array, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return Image.fromarray(array, mode="RGB")


def get_elements_dataset(
    data_paths: Sequence[str],
    normalize_data: bool = True,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    element_count: int = DEFAULT_ELEMENT_COUNT,
    element_size: int = DEFAULT_ELEMENT_SIZE,
    element_size_delta: int = DEFAULT_ELEMENT_SIZE_DELTA,
    element_seed: int = DEFAULT_ELEMENT_SEED,
    loc_seed: int = DEFAULT_LOC_SEED,
    allowed_shapes: Optional[Sequence[str]] = None,
    allowed_colors: Optional[Sequence[str]] = None,
    allowed_textures: Optional[Sequence[str]] = None,
    class_configs: Optional[Sequence[Dict[str, Optional[str]]]] = None,
    concept_names: Optional[Sequence[str]] = None,
    allowed_combinations: Optional[Sequence[Sequence[str]]] = None,
    place_remaining_randomly: bool = DEFAULT_PLACE_REMAINING_RANDOMLY,
    correlations: Optional[Sequence[Dict[str, object]]] = None,
    augmentation: Optional[T.Compose] = None,
    artifact_ids_file: Optional[str] = None,
    **dataset_kwargs,
) -> "ElementsDataset":
    transform = _normalization_transform(normalize_data, image_size)
    if augmentation is None:
        augmentation = elements_augmentation

    return ElementsDataset(
        data_paths=data_paths,
        transform=transform,
        augmentation=augmentation,
        artifact_ids_file=artifact_ids_file,
        num_samples=num_samples,
        image_size=image_size,
        element_count=element_count,
        element_size=element_size,
        element_size_delta=element_size_delta,
        element_seed=element_seed,
        loc_seed=loc_seed,
        allowed_shapes=allowed_shapes,
        allowed_colors=allowed_colors,
        allowed_textures=allowed_textures,
        class_configs=class_configs,
        concept_names=concept_names,
        allowed_combinations=allowed_combinations,
        place_remaining_randomly=place_remaining_randomly,
        correlations=correlations,
        **dataset_kwargs,
    )


class ElementsDataset(BaseDataset):
    """Adapter that exposes the procedural ElementDataset via BaseDataset methods."""

    def __init__(
        self,
        data_paths: Sequence[str],
        transform: Optional[T.Compose],
        augmentation: Optional[T.Compose],
        artifact_ids_file: Optional[str],
        num_samples: int,
        image_size: int,
        element_count: int,
        element_size: int,
        element_size_delta: int,
        element_seed: int,
        loc_seed: int,
        allowed_shapes: Optional[Sequence[str]],
        allowed_colors: Optional[Sequence[str]],
        allowed_textures: Optional[Sequence[str]],
        class_configs: Optional[Sequence[Dict[str, Optional[str]]]],
        concept_names: Optional[Sequence[str]],
        allowed_combinations: Optional[Sequence[Sequence[str]]],
        place_remaining_randomly: bool,
        correlations: Optional[Sequence[Dict[str, object]]],
        **extra_kwargs,
    ) -> None:
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)

        self.num_samples = int(num_samples)
        self.img_size = int(image_size)
        self.element_count = int(element_count)
        self.element_size = int(element_size)
        self.element_size_delta = int(element_size_delta)
        self.element_seed = int(element_seed)
        self.loc_seed = int(loc_seed)
        self.place_remaining_randomly = place_remaining_randomly

        self.allowed_shapes = tuple(allowed_shapes) if allowed_shapes else DEFAULT_ALLOWED_SHAPES
        self.allowed_colors = tuple(allowed_colors) if allowed_colors else DEFAULT_ALLOWED_COLORS
        self.allowed_textures = tuple(allowed_textures) if allowed_textures else DEFAULT_ALLOWED_TEXTURES

        default_concept_configs, default_concept_names = _default_class_configs(
            self.allowed_shapes, self.allowed_colors, self.allowed_textures
        )

        self.concept_configs = list(default_concept_configs)
        if concept_names is None:
            self.concept_names = list(default_concept_names)
        else:
            if len(concept_names) != len(self.concept_configs):
                raise ValueError("Length of concept_names must match concept configurations.")
            self.concept_names = list(concept_names)

        if class_configs is not None:
            self.class_configs = [dict(cfg) for cfg in class_configs]
            self.class_names = [_compose_class_name(idx, class_config) for idx, class_config in enumerate(self.class_configs)]
        else:
            self.class_configs = [dict(cfg) for cfg in self.concept_configs]
            self.class_names = list(self.concept_names)

        self.allowed = _build_allowed_dict(
            self.allowed_shapes,
            self.allowed_colors,
            self.allowed_textures,
            self.element_size,
            self.element_size_delta,
        )

        combinations = None
        if allowed_combinations is not None:
            combinations = [tuple(combo) for combo in allowed_combinations]

        generator_kwargs = dict(
            allowed=self.allowed,
            class_configs=self.concept_configs,
            n=self.num_samples,
            img_size=self.img_size,
            element_n=self.element_count,
            element_size=self.element_size,
            element_size_delta=self.element_size_delta,
            element_seed=self.element_seed,
            loc_seed=self.loc_seed,
            allowed_combinations=combinations,
            loc_restrictions=None,
            place_remaining_randomly=self.place_remaining_randomly,
        )

        if extra_kwargs:
            log.debug("Ignoring unsupported kwargs for ElementsDataset: %s", sorted(extra_kwargs.keys()))

        self._generator = ElementDataset(**generator_kwargs) # type: ignore

        self.sample_names: List[str] = [f"element_{idx:05d}" for idx in range(self.num_samples)]
        self.metadata = pd.DataFrame(
            {
                "sample_id": self.sample_names,
                "element_seed": self._generator.element_seeds.astype(int),
                "loc_seed": self._generator.loc_seeds.astype(int),
            }
        )

        labels: List[torch.Tensor] = []
        class_labels: List[torch.Tensor] = []
        self._element_info: List[Dict] = []
        for idx in range(self.num_samples):
            element_image = self._generator.get_item(idx)
            labels.append(torch.tensor(np.asarray(element_image.class_labels_oh), dtype=torch.float32))
            class_labels.append(self._compute_class_vector(element_image))
            self._element_info.append(element_image.info)
        self.concept_labels = torch.stack(labels) if labels else torch.zeros((0, 0), dtype=torch.float32)
        self.labels = torch.stack(class_labels) if class_labels else torch.zeros((0, 0), dtype=torch.float32)

        if self.concept_labels.ndim == 1:
            self.concept_labels = self.concept_labels.unsqueeze(1)
        if self.labels.ndim == 1:
            self.labels = self.labels.unsqueeze(1)

        self._apply_correlations(correlations)

        self.mean = torch.tensor(DEFAULT_MEAN)
        self.var = torch.tensor(DEFAULT_STD)
        self.normalize_fn = T.Normalize(DEFAULT_MEAN, DEFAULT_STD)

        class_counts = _compute_concept_counts(self.labels)
        self.weights = self.compute_weights(class_counts) if class_counts.size else torch.tensor([])
        self._refresh_index_maps()

    def __len__(self) -> int:
        return self.num_samples

    def _compute_class_vector(self, element_image) -> torch.Tensor:
        if not self.class_configs:
            return torch.zeros((0,), dtype=torch.float32)
        vec = torch.zeros(len(self.class_configs), dtype=torch.float32)
        for class_idx, config in enumerate(self.class_configs):
            if element_image.belongs_to_class(config):
                vec[class_idx] = 1.0
        return vec

    def _refresh_index_maps(self) -> None:
        self.sample_ids_by_concept = {
            name: torch.nonzero(self.concept_labels[:, idx], as_tuple=False).squeeze(1).cpu().numpy()
            for idx, name in enumerate(self.concept_names)
        } if self.concept_labels.numel() else {name: np.array([], dtype=int) for name in self.concept_names}
        self.sample_ids_by_class = {
            name: torch.nonzero(self.labels[:, idx], as_tuple=False).squeeze(1).cpu().numpy()
            for idx, name in enumerate(self.class_names)
        } if self.labels.numel() else {name: np.array([], dtype=int) for name in self.class_names}

    def __getitem__(self, idx: int):
        element_image = self._generator.get_item(idx)
        image = _pil_from_element_image(element_image)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        if self.augmentation and self.do_augmentation:
            image = self.augmentation(image)

        target = self.labels[idx]
        return image.float(), target.clone()

    def get_all_ids(self) -> List[str]:
        return list(self.sample_names)

    def get_sample_name(self, i: int) -> str:
        return self.sample_names[i]

    def get_target(self, i: int) -> torch.Tensor:
        return self.labels[i]

    def get_labels(self) -> torch.Tensor:
        return self.labels.clone()

    def get_concept_names(self) -> List[str]:
        return list(self.concept_names)

    def get_class_names(self) -> List[str]:
        return list(self.class_names)

    def get_concept_labels(self) -> torch.Tensor:
        return self.concept_labels.clone()

    def get_class_labels(self) -> torch.Tensor:
        return self.labels.clone()

    def get_subset_by_idxs(self, idxs: Sequence[int]) -> "ElementsDataset":
        subset = copy.deepcopy(self)
        subset.sample_names = [self.sample_names[i] for i in idxs]
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True) # type: ignore
        subset.labels = self.labels[idxs].clone()
        subset.concept_labels = self.concept_labels[idxs].clone()
        subset.num_samples = len(idxs)
        subset._element_info = [self._element_info[i] for i in idxs]
        subset._refresh_index_maps()
        return subset

    def _apply_correlations(self, correlations: Optional[Sequence[Dict[str, object]]]) -> None:
        if not correlations:
            return

        rng = np.random.default_rng(self.element_seed + self.loc_seed + 1337)
        for entry in correlations:
            if entry is None:
                continue
            indices = entry.get("concept_indices") or entry.get("indices") or entry.get("pair")
            if indices is None and entry.get("concept_names") is not None:
                names = entry.get("concept_names")
                if isinstance(names, (str, bytes)):
                    names = [names]
                try:
                    indices = [self.concept_names.index(name) for name in names] # type: ignore
                except ValueError as err:
                    warnings.warn(f"Correlation request skipped: {err}")
                    continue
            if indices is None:
                warnings.warn("Correlation entry missing concept indices; skipping.")
                continue
            if len(indices) != 2:   # type: ignore
                warnings.warn(f"Correlation entry requires exactly 2 indices, got {indices}; skipping.")
                continue
            idx_a, idx_b = (int(indices[0]), int(indices[1])) # type: ignore
            if not (0 <= idx_a < self.concept_labels.shape[1] and 0 <= idx_b < self.concept_labels.shape[1]):
                warnings.warn(f"Correlation indices {(idx_a, idx_b)} out of range; skipping.")
                continue
            degree = float(entry.get("degree", 0.0)) # type: ignore
            degree = float(np.clip(degree, 0.0, 1.0))
            if degree in (0.0, 1.0):
                log.debug("Enforcing correlation degree %.2f between concepts %s and %s", degree, self.concept_names[idx_a], self.concept_names[idx_b])
            trigger_indices = torch.nonzero(self.concept_labels[:, idx_a] > 0, as_tuple=False).squeeze(1).tolist()
            for sample_idx in trigger_indices:
                desired = 1 if rng.random() < degree else 0
                current = int(round(self.concept_labels[sample_idx, idx_b].item()))
                if current == desired:
                    continue
                self._resample_sample(sample_idx, idx_a, idx_b, desired, rng)
        self._refresh_index_maps()

    def _resample_sample(self, sample_idx: int, trigger_idx: int, target_idx: int, desired: int, rng: np.random.Generator, max_attempts: int = 100) -> None:
        original_element_seed = int(self._generator.element_seeds[sample_idx]) # type: ignore
        original_loc_seed = int(self._generator.loc_seeds[sample_idx]) # type: ignore
        original_concept_labels = self.concept_labels[sample_idx].clone()
        original_class_labels = self.labels[sample_idx].clone()
        original_info = self._element_info[sample_idx]

        for _ in range(max_attempts):
            self._generator.element_seeds[sample_idx] = rng.integers(0, 2**32, dtype=np.uint32) # type: ignore
            self._generator.loc_seeds[sample_idx] = rng.integers(0, 2**32, dtype=np.uint32) # type: ignore
            element_image = self._generator.get_item(sample_idx)
            labels_np = np.asarray(element_image.class_labels_oh)
            labels = torch.tensor(labels_np, dtype=torch.float32)
            if labels.ndim == 1:
                pass
            if labels[trigger_idx] > 0 and int(round(labels[target_idx].item())) == desired:
                self.concept_labels[sample_idx] = labels
                class_vec = self._compute_class_vector(element_image)
                self.labels[sample_idx] = class_vec
                self._element_info[sample_idx] = element_image.info
                self.metadata.at[sample_idx, "element_seed"] = int(self._generator.element_seeds[sample_idx]) # type: ignore
                self.metadata.at[sample_idx, "loc_seed"] = int(self._generator.loc_seeds[sample_idx]) # type: ignore
                return

        # Revert if unsuccessful
        self._generator.element_seeds[sample_idx] = original_element_seed # type: ignore
        self._generator.loc_seeds[sample_idx] = original_loc_seed # type: ignore
        self.concept_labels[sample_idx] = original_concept_labels
        self.labels[sample_idx] = original_class_labels
        self._element_info[sample_idx] = original_info
        self.metadata.at[sample_idx, "element_seed"] = original_element_seed
        self.metadata.at[sample_idx, "loc_seed"] = original_loc_seed
        warnings.warn(
            f"Unable to satisfy correlation constraint for sample {sample_idx} (trigger={self.concept_names[trigger_idx]}, target={self.concept_names[target_idx]}, desired={desired})."
        )

    def get_num_classes(self) -> int:
        return len(self.class_names)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    correlations = [
        {"concept_names": ["shape:triangle", "texture:stripes_horizontal"], "degree": 0.8}
    ]

    dataset = get_elements_dataset(data_paths=["."], num_samples=512, correlations=correlations)
    dataset.do_augmentation = False

    n_show = min(9, len(dataset))
    n_cols = min(3, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(np.array(axes).reshape(-1))

    for idx in range(n_show):
        image_tensor, label_vec = dataset[idx]
        image_uint8 = dataset.reverse_normalization(image_tensor).permute(1, 2, 0).cpu().numpy()
        image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)

        class_ids = torch.nonzero(label_vec > 0, as_tuple=False).squeeze(1).tolist()
        active_classes = [dataset.class_names[c] for c in class_ids] if class_ids else ["no classes"]

        axes[idx].imshow(image_uint8)
        axes[idx].set_title("\n".join(active_classes), fontsize=10)
        axes[idx].axis("off")

    for ax in axes[n_show:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()

    print(f"Concept names: {dataset.get_concept_names()}")
    print(f"Class names: {dataset.get_class_names()}")

    labels_tensor = dataset.get_labels()
    labels_np = labels_tensor.numpy()
    if labels_np.ndim == 1:
        labels_np = labels_np[:, None]
    corr_matrix = np.corrcoef(labels_np, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    corr_tensor = torch.tensor(corr_matrix, dtype=torch.float32)
    visualize_sim_matrix(corr_tensor, dataset.get_concept_names(), title="Concept Correlation Matrix", display=True)
    
