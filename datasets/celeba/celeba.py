import copy
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import CelebA
from datasets.base_dataset import BaseDataset
import logging

log = logging.getLogger(__name__)

celeba_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.5),
    # T.RandomVerticalFlip(p=.5),
    # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25),
    T.RandomApply(transforms=[T.Pad(10, fill=0), T.Resize(224)], p=.25)
])


def get_celeba_dataset(data_paths, normalize_data=True, image_size=224, artifact_ids_file=None, **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebADataset(data_paths, transform=transform, augmentation=celeba_augmentation,
                         artifact_ids_file=artifact_ids_file)


class CelebADataset(BaseDataset):
    def __init__(self, data_paths, transform=None, augmentation=None, artifact_ids_file=None):
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)
        assert len(data_paths) == 1, "Only 1 path accepted for CelebA Dataset"

        ds = CelebA(root=data_paths[0], split='all', download=False, transform=transform)
        self.path = f"{data_paths[0]}/{ds.base_folder}"
        
        self.attributes = pd.DataFrame(ds.attr, columns=ds.attr_names[:-1]) # type: ignore
        
        USE_SUBSET = True
        if USE_SUBSET:
            log.info("Using subset.")
            NTH = 10
        else:
            NTH = 1
        filter_indices = np.zeros(len(ds.attr))
        filter_indices[::NTH] = 1
        
        self.attributes = self.attributes[filter_indices == 1].reset_index(drop=True)
        
        self.sample_ids_by_concept = {}
        for attr in self.attributes.columns:
            self.sample_ids_by_concept[attr] = np.where(self.attributes[attr].values == 1)[0]
            
        ATTR_LABEL = "Blond_Hair"
        labels = self.attributes[ATTR_LABEL].values
        
        self.metadata = pd.DataFrame(
            {'image_id': np.array(ds.filename)[filter_indices == 1], 'targets': labels})
        
        self.normalize_fn = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.classes = [f'Non-{ATTR_LABEL}', ATTR_LABEL]
        self.class_names = self.classes

        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.var = torch.Tensor([0.5, 0.5, 0.5])

        self.weights = self.compute_weights(np.array([len(labels) - labels.sum(), labels.sum()]))
        
    def get_all_ids(self):
        return list(self.metadata['image_id'].values)
    
    def get_labels(self):
        return torch.tensor(self.attributes.to_numpy(), dtype=torch.float32)
    
    def get_concept_names(self):
        return list(self.sample_ids_by_concept.keys())

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")

        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image) # type: ignore

        return image.float(), target # type: ignore

    def get_sample_name(self, i):
        return self.metadata.iloc[i]['image_id']

    def get_target(self, i):
        target = torch.tensor(self.metadata.iloc[i]["targets"])
        return target

    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset


if __name__ == "__main__":
    data_paths = ["/Users/erogullari/datasets/"]
    ds = get_celeba_dataset(data_paths, normalize_data=True, image_size=224)
    print(len(ds))
