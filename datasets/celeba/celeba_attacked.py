import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets.celeba.celeba_subset import CelebASubset, celeba_augmentation
from datasets.celeba.artificial_artifact import insert_artifact
import logging

log = logging.getLogger(__name__)

ARTIFACT_CONCEPT_POOL = [
    "timestamp",
    "box",
    "brightness",
    "checkerboard",
    "watermark",
]


def get_active_artifact_concepts(num_concepts):
    if isinstance(num_concepts, bool) or not isinstance(num_concepts, (int, np.integer)):
        raise TypeError("num_concepts must be an integer.")
    if not 1 <= int(num_concepts) <= len(ARTIFACT_CONCEPT_POOL):
        raise ValueError(
            f"num_concepts must be between 1 and {len(ARTIFACT_CONCEPT_POOL)}, "
            f"got {num_concepts}."
        )
    return ARTIFACT_CONCEPT_POOL[: int(num_concepts)]


def generate_artifact_labels(
    targets,
    attacked_classes,
    num_concepts=3,
    p_artifact=0.5,
    cooccurrence_probability=0.5,
    entanglement_factor=5,
    artifact_seed=0,
):
    concepts = get_active_artifact_concepts(num_concepts)
    if not 0 <= p_artifact <= 1:
        raise ValueError("p_artifact must be between 0 and 1.")
    if not 0 <= cooccurrence_probability <= 1:
        raise ValueError("cooccurrence_probability must be between 0 and 1.")
    if entanglement_factor <= 0:
        raise ValueError("entanglement_factor must be positive.")

    targets = np.asarray(targets)
    attacked_classes = set(attacked_classes)
    rng = np.random.RandomState(artifact_seed)
    timestamp_probabilities = np.where(
        np.isin(targets, list(attacked_classes)), p_artifact, 0.005
    )
    labels = {
        "timestamp": rng.rand(len(targets)) < timestamp_probabilities,
    }
    p_without_timestamp = cooccurrence_probability / entanglement_factor
    for concept in concepts[1:]:
        conditional_probabilities = np.where(
            labels["timestamp"], cooccurrence_probability, p_without_timestamp
        )
        labels[concept] = rng.rand(len(targets)) < conditional_probabilities
    return labels


def get_celeba_attacked_dataset(
    data_paths,
    normalize_data=True,
    image_size=224,
    attacked_classes=[],
    p_artifact=0.5,
    artifact_type="ch_text",
    num_concepts=3,
    cooccurrence_probability=0.5,
    entanglement_factor=5,
    artifact_seed=0,
    **kwargs,
):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebAAttackedDataset(
        data_paths,
        transform=transform,
        augmentation=celeba_augmentation,
        attacked_classes=attacked_classes,
        p_artifact=p_artifact,
        artifact_type=artifact_type,
        image_size=image_size,
        num_concepts=num_concepts,
        cooccurrence_probability=cooccurrence_probability,
        entanglement_factor=entanglement_factor,
        artifact_seed=artifact_seed,
        **kwargs,
    )


class CelebAAttackedDataset(CelebASubset):
    def __init__(
        self,
        data_paths,
        transform=None,
        augmentation=None,
        attacked_classes=[],
        p_artifact=0.2,
        artifact_type="ch_text",
        image_size=224,
        num_concepts=3,
        cooccurrence_probability=0.5,
        entanglement_factor=5,
        artifact_seed=0,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        **artifact_kwargs,
    ):
        super().__init__(
            data_paths, transform, augmentation, None, val_split, test_split, seed
        )

        self.image_size = image_size
        self.transform_resize = T.Resize(
            (image_size, image_size), interpolation=T.InterpolationMode.BICUBIC
        )
        self.num_concepts = int(num_concepts)
        self.artifact_concepts = get_active_artifact_concepts(num_concepts)
        self.artifact_seed = int(artifact_seed)
        timestamp_kwargs = dict(artifact_kwargs)
        timestamp_kwargs.setdefault("img_size", image_size)
        brightness_kwargs = {
            "factor": artifact_kwargs.get(
                "brightness_factor", artifact_kwargs.get("factor", 1.4)
            )
        }
        all_artifact_specs = {
            "timestamp": (artifact_type, timestamp_kwargs),
            "box": ("random_box", {}),
            "brightness": ("brightness", brightness_kwargs),
            "checkerboard": ("checkerboard", {}),
            "watermark": ("watermark", {}),
        }
        self.artifact_specs = {
            concept: all_artifact_specs[concept]
            for concept in self.artifact_concepts
        }
        self.artifact_labels_by_concept = generate_artifact_labels(
            self.metadata.targets.to_numpy(),
            attacked_classes=attacked_classes,
            num_concepts=num_concepts,
            p_artifact=p_artifact,
            cooccurrence_probability=cooccurrence_probability,
            entanglement_factor=entanglement_factor,
            artifact_seed=artifact_seed,
        )

        self._refresh_artifact_indices()

        for concept, sample_ids in self.sample_ids_by_artifact.items():
            log.info(f"Adding concept {concept} into metadata.")
            self.metadata[concept] = 0
            self.metadata.loc[sample_ids, concept] = 1  # type: ignore

        log.info(
            "Inserted artifacts: %s",
            " / ".join(
                f"{concept} ({int(labels.sum())})"
                for concept, labels in self.artifact_labels_by_concept.items()
            ),
        )

    def _refresh_artifact_indices(self):
        self.sample_ids_by_artifact = {
            concept: np.where(labels)[0]
            for concept, labels in self.artifact_labels_by_concept.items()
        }
        artifact_ids = list(self.sample_ids_by_artifact.values())
        self.artifact_ids_union = (
            np.unique(np.concatenate(artifact_ids))
            if artifact_ids
            else np.array([], dtype=int)
        )
        artifact_id_set = set(self.artifact_ids_union.tolist())
        self.clean_sample_ids = [
            i for i in range(len(self)) if i not in artifact_id_set
        ]

    def add_artifact(
        self,
        img,
        idx,
        artifact_type,
        concept=None,
        occupied_mask=None,
        **artifact_kwargs,
    ):
        if concept is None:
            concept = next(
                (
                    name
                    for name, (candidate_type, _) in self.artifact_specs.items()
                    if candidate_type == artifact_type
                ),
                artifact_type,
            )
        concept_offset = ARTIFACT_CONCEPT_POOL.index(concept) + 1
        insertion_seed = (
            self.artifact_seed * 1_000_003 + int(idx) * 101 + concept_offset * 10_007
        ) % (2**32)

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(insertion_seed)
            np.random.seed(insertion_seed)
            torch.manual_seed(insertion_seed)
            if occupied_mask is not None:
                artifact_kwargs["occupied_mask"] = occupied_mask
            return insert_artifact(img, artifact_type, **artifact_kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def render_artifacts(self, image, idx):
        masks = {
            concept: torch.zeros((self.image_size, self.image_size)).float()
            for concept in self.artifact_concepts
        }

        # Global transformations must happen before localized overlays.
        if (
            "brightness" in self.artifact_concepts
            and self.artifact_labels_by_concept["brightness"][idx]
        ):
            artifact_type, kwargs = self.artifact_specs["brightness"]
            image, mask = self.add_artifact(
                image, idx, artifact_type, concept="brightness", **kwargs
            )
            masks["brightness"] = mask.float()

        occupied_mask = torch.zeros((self.image_size, self.image_size)).bool()
        for concept in self.artifact_concepts:
            if concept == "brightness" or not self.artifact_labels_by_concept[concept][idx]:
                continue
            artifact_type, kwargs = self.artifact_specs[concept]
            image, mask = self.add_artifact(
                image,
                idx,
                artifact_type,
                concept=concept,
                occupied_mask=occupied_mask,
                **kwargs,
            )
            masks[concept] = mask.float()
            occupied_mask |= mask.bool()
        return image, masks

    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")
        image = self.transform_resize(image)
        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        image, _ = self.render_artifacts(image, idx)

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)  # type: ignore

        return image.float(), target  # type: ignore

    # Overrides
    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.attributes = self.attributes.iloc[idxs].reset_index(drop=True)
        subset.sample_ids_by_concept = {
            concept: np.where(subset.attributes[concept].to_numpy() == 1)[0]
            for concept in subset.attributes.columns
        }
        subset.artifact_labels_by_concept = {
            concept: labels[np.array(idxs)]
            for concept, labels in self.artifact_labels_by_concept.items()
        }
        subset._refresh_artifact_indices()
        return subset

    def get_labels(self):
        base_labels = super().get_labels()
        attack_cols = torch.tensor(
            self.metadata[self.artifact_concepts].to_numpy(),
            dtype=base_labels.dtype,
        )
        return torch.cat([base_labels, attack_cols], dim=1)

    def get_concept_names(self):
        concept_names = super().get_concept_names()
        if "timestamp" not in concept_names:
            concept_names = concept_names + self.artifact_concepts
        return concept_names


if __name__ == "__main__":
    import torchvision

    logging.basicConfig(level=logging.INFO)
    data_paths = ["/Users/erogullari/datasets/"]
    ds = get_celeba_attacked_dataset(
        data_paths,
        normalize_data=True,
        image_size=224,
        attacked_classes=[1],
        p_artifact=1,
        artifact_type="ch_time",
        entanglement_factor=10,
    )
    for i in range(20):
        img, _ = ds[i]
        torchvision.utils.save_image(
            ds.reverse_normalization(img).float() / 255.0,
            f"DELETE/celeba_attacked/celeba_sample{i}.png",
        )
