import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets.celeba.celeba_subset import CelebASubset, celeba_augmentation
from datasets.celeba.artificial_artifact import insert_artifact
import logging

log = logging.getLogger(__name__)

ARTIFACT_CONCEPTS = ["timestamp", "box", "brightness"]


def get_celeba_attacked_dataset(
    data_paths,
    normalize_data=True,
    image_size=224,
    attacked_classes=[],
    p_artifact=0.5,
    artifact_type="ch_text",
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
        **kwargs,
    )


class CelebAAttackedDataset(CelebASubset):
    artifact_concepts = ARTIFACT_CONCEPTS

    def __init__(
        self,
        data_paths,
        transform=None,
        augmentation=None,
        attacked_classes=[],
        p_artifact=0.2,
        artifact_type="ch_text",
        image_size=224,
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

        ## art1 dependant on target (spurious correlation)
        p_art1_base = 0.005
        p_art1 = {
            cl: p_artifact if cl in attacked_classes else p_art1_base
            for cl in self.metadata.targets.drop_duplicates().values
        }

        ## art2 dependant on art1 (correlated feature)
        base_prob = 0.5
        entanglement_factor = artifact_kwargs["entanglement_factor"]
        p_art2 = {0: base_prob / entanglement_factor, 1: base_prob}
        ## art3 dependant on art1 only (correlated brightness)
        p_art3 = {0: base_prob / entanglement_factor, 1: base_prob}

        self.art1_type = artifact_type
        self.art2_type = "random_box"
        self.art3_type = "brightness"

        self.art1_kwargs = artifact_kwargs
        self.art2_kwargs = {}
        self.art3_kwargs = {
            "factor": artifact_kwargs.get(
                "brightness_factor", artifact_kwargs.get("factor", 1.4)
            )
        }
        self.artifact_specs = {
            "timestamp": (self.art1_type, self.art1_kwargs),
            "box": (self.art2_type, self.art2_kwargs),
            "brightness": (self.art3_type, self.art3_kwargs),
        }

        np.random.seed(0)
        self.art1_labels = np.array(
            [
                np.random.rand() < p_art1[self.metadata.iloc[i].targets]
                for i in range(len(self))
            ]
        )

        self.art2_labels = np.array(
            [
                np.random.rand() < p_art2[int(self.art1_labels[i])]
                for i in range(len(self))
            ]
        )

        self.art3_labels = np.array(
            [
                np.random.rand() < p_art3[int(self.art1_labels[i])]
                for i in range(len(self))
            ]
        )

        self._refresh_artifact_indices()

        for concept, sample_ids in self.sample_ids_by_artifact.items():
            log.info(f"Adding concept {concept} into metadata.")
            self.metadata[concept] = 0
            self.metadata.loc[sample_ids, concept] = 1  # type: ignore

        log.info(
            "Inserted artifacts: timestamp (%s) / box (%s) / brightness (%s)",
            self.art1_labels.sum(),
            self.art2_labels.sum(),
            self.art3_labels.sum(),
        )

    def _refresh_artifact_indices(self):
        self.artifact_labels_by_concept = {
            "timestamp": self.art1_labels,
            "box": self.art2_labels,
            "brightness": self.art3_labels,
        }
        self.art1_ids = np.where(self.art1_labels)[0]
        self.art2_ids = np.where(self.art2_labels)[0]
        self.art3_ids = np.where(self.art3_labels)[0]
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

    def add_artifact(self, img, idx, artifact_type, **artifact_kwargs):
        random.seed(idx)
        torch.manual_seed(idx)
        np.random.seed(idx)

        return insert_artifact(img, artifact_type, **artifact_kwargs)

    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")
        image = self.transform_resize(image)
        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        for concept in self.artifact_concepts:
            if self.artifact_labels_by_concept[concept][idx]:
                artifact_type, kwargs = self.artifact_specs[concept]
                image, _ = self.add_artifact(image, idx, artifact_type, **kwargs)

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)  # type: ignore

        return image.float(), target  # type: ignore

    # Overrides
    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.art1_labels = self.art1_labels[np.array(idxs)]
        subset.art2_labels = self.art2_labels[np.array(idxs)]
        subset.art3_labels = self.art3_labels[np.array(idxs)]

        subset._refresh_artifact_indices()
        return subset

    def get_labels(self):
        base_labels = super().get_labels()  # shape [N, 40]
        attack_cols = torch.tensor(
            self.metadata[self.artifact_concepts].to_numpy(),
            dtype=base_labels.dtype,
        )
        return torch.cat([base_labels, attack_cols], dim=1)  # shape [N, 43]

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
