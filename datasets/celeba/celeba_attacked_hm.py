import torch
import torchvision.transforms as T
from PIL import Image
from datasets.celeba.celeba import CelebADataset, celeba_augmentation
from datasets.celeba.celeba_attacked import CelebAAttackedDataset
import logging

log = logging.getLogger(__name__)

def get_celeba_attacked_hm_dataset(data_paths, normalize_data=True, image_size=224, attacked_classes=[], 
                                   p_artifact=.5, artifact_type='ch_text', **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebAAttackedHmDataset(data_paths, transform=transform, augmentation=celeba_augmentation, 
                                 attacked_classes=attacked_classes, p_artifact=p_artifact, 
                                 artifact_type=artifact_type, image_size=image_size, **kwargs)

class CelebAAttackedHmDataset(CelebAAttackedDataset):
    def __init__(self, 
                 data_paths, 
                 transform=None, 
                 augmentation=None, 
                 attacked_classes=[],
                 p_artifact=.2,
                 artifact_type="ch_text",
                 image_size=224,
                 **artifact_kwargs):
        super().__init__(data_paths, transform, augmentation, attacked_classes,
                         p_artifact, artifact_type, image_size, **artifact_kwargs)
    
    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")
        image = self.transform_resize(image)
        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        # Art1
        if self.art1_labels[idx]:
            image, mask_art1 = self.add_artifact(image, idx, self.art1_type, **self.art1_kwargs)
        else:
            mask_art1 = torch.zeros((self.image_size, self.image_size)).float()

        # Art2
        if self.art2_labels[idx]:
            image, mask_art2 = self.add_artifact(image, idx, self.art2_type, **self.art2_kwargs)
        else:
            mask_art2 = torch.zeros((self.image_size, self.image_size)).float()

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)    # type: ignore

        return image.float(), target, mask_art1, mask_art2  # type: ignore
