import torch
import torchvision.transforms as T
from PIL import Image
from datasets.celeba.celeba import celeba_augmentation
from datasets.celeba.celeba_attacked import CelebAAttackedDataset
import logging

log = logging.getLogger(__name__)

def get_celeba_attacked_hm_dataset(
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
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebAAttackedHmDataset(
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

class CelebAAttackedHmDataset(CelebAAttackedDataset):
    def __init__(self, 
                 data_paths, 
                 transform=None, 
                 augmentation=None, 
                 attacked_classes=[],
                 p_artifact=.2,
                 artifact_type="ch_text",
                 image_size=224,
                 num_concepts=3,
                 cooccurrence_probability=0.5,
                 entanglement_factor=5,
                 artifact_seed=0,
                 **artifact_kwargs):
        super().__init__(
            data_paths,
            transform,
            augmentation,
            attacked_classes,
            p_artifact,
            artifact_type,
            image_size,
            num_concepts,
            cooccurrence_probability,
            entanglement_factor,
            artifact_seed,
            **artifact_kwargs,
        )
    
    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")
        image = self.transform_resize(image)
        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        image, artifact_masks = self.render_artifacts(image, idx)

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)    # type: ignore

        return image.float(), target, artifact_masks  # type: ignore
