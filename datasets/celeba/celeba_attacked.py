import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.celeba.celeba import CelebADataset, celeba_augmentation
from datasets.celeba.artificial_artifact import insert_artifact
import logging

log = logging.getLogger(__name__)

def get_celeba_attacked_dataset(data_paths, normalize_data=True, image_size=224, attacked_classes=[], 
                       p_artifact=.5, artifact_type='ch_text', **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebAAttackedDataset(data_paths, transform=transform, augmentation=celeba_augmentation, 
                                 attacked_classes=attacked_classes, p_artifact=p_artifact, 
                                 artifact_type=artifact_type, image_size=image_size, **kwargs)

class CelebAAttackedDataset(CelebADataset):
    def __init__(self, 
                 data_paths, 
                 transform=None, 
                 augmentation=None, 
                 attacked_classes=[],
                 p_artifact=.2,
                 artifact_type="ch_text",
                 image_size=224,
                 **artifact_kwargs):
        super().__init__(data_paths, transform, augmentation, None)

        self.image_size = image_size
        self.transform_resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)

        ## art1 dependant on target (spurious correlation)
        p_art1_base = 0.005
        p_art1 = {cl: p_artifact if cl in attacked_classes else p_art1_base 
                  for cl in self.metadata.targets.drop_duplicates().values}

        ## art2 dependant on art1 (correlated feature)
        base_prob = .5
        entanglement_factor = artifact_kwargs["entanglement_factor"]
        p_art2 = {0: base_prob / entanglement_factor, 
                  1: base_prob}

        self.art1_type = artifact_type
        self.art2_type = "random_box"

        self.art1_kwargs = artifact_kwargs
        self.art2_kwargs = {}

        np.random.seed(0)
        self.art1_labels = np.array([
            np.random.rand() < p_art1[self.metadata.iloc[i].targets] 
            for i in range(len(self))])
        
        self.art2_labels = np.array([
            np.random.rand() < p_art2[int(self.art1_labels[i])] 
            for i in range(len(self))])
    
        self.art1_ids = np.where(self.art1_labels)[0]
        self.art2_ids = np.where(self.art2_labels)[0]
        self.sample_ids_by_artifact = {"timestamp": self.art1_ids, 
                                       "box": self.art2_ids}
        
        self.artifact_ids_union = np.union1d(self.art1_ids, self.art2_ids)
        self.clean_sample_ids = [i for i in range(len(self)) if i not in self.artifact_ids_union]

        for concept, sample_ids in self.sample_ids_by_artifact.items():
            log.info(f"Adding concept {concept}")
            self.metadata[concept] = 0
            self.metadata.loc[sample_ids, concept] = 1      # type: ignore
            
        log.info(f"Inserting artifacts: {self.art1_labels.sum()} / {self.art2_labels.sum()}")

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

        # Art1
        if self.art1_labels[idx]:
            image, _ = self.add_artifact(image, idx, self.art1_type, **self.art1_kwargs)

        # Art2
        if self.art2_labels[idx]:
            image, _ = self.add_artifact(image, idx, self.art2_type, **self.art2_kwargs)

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image) # type: ignore

        return image.float(), target # type: ignore


    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.art1_labels = self.art1_labels[np.array(idxs)]
        subset.art2_labels = self.art2_labels[np.array(idxs)]

        subset.art1_ids = np.where(subset.art1_labels)[0]
        subset.art2_ids = np.where(subset.art2_labels)[0]
        subset.sample_ids_by_artifact = {"timestamp": subset.art1_ids, 
                                         "box": subset.art2_ids}
        subset.artifact_ids_union = np.union1d(subset.art1_ids, subset.art2_ids)
        subset.clean_sample_ids = [i for i in range(len(subset)) if i not in subset.artifact_ids_union]
        return subset

if __name__ == "__main__":
    import torchvision
    logging.basicConfig(level=logging.INFO)
    data_paths = ["/Users/erogullari/datasets/"]
    ds = get_celeba_attacked_dataset(data_paths, 
                                     normalize_data=True, 
                                     image_size=224, 
                                     attacked_classes=[1], 
                                     p_artifact=1, 
                                     artifact_type="ch_time",
                                     entanglement_factor=10
                                    )
    for i in range(20):
        img, _ = ds[i]
        torchvision.utils.save_image(ds.reverse_normalization(img).float() / 255.0, f"DELETE/celeba_attacked/celeba_sample{i}.png")