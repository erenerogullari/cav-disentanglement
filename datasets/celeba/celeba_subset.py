import copy

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import CelebA
import logging
from datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)

NORM_PARAMS_CELEBA = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
FN_NORMALIZE_CELEBA = T.Normalize(*NORM_PARAMS_CELEBA)

celeba_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.5),
    # T.RandomVerticalFlip(p=.5),
    # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25),
    T.RandomApply(transforms=[T.Pad(10, fill=-(46.9 / 255.) / (22.6 / 255.)), T.Resize(224)], p=.25)
])


def get_celeba_subset(data_paths, normalize_data=True, image_size=224, artifact_ids_file=None, **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(FN_NORMALIZE_CELEBA)

    transform = T.Compose(fns_transform)

    return CelebASubset(data_paths, transform=transform, augmentation=celeba_augmentation,
                         artifact_ids_file=artifact_ids_file, **kwargs)


class CelebASubset(BaseDataset):
    classes = [0,1]
    
    def __init__(self, data_paths, transform=None, augmentation=None, artifact_ids_file=None, val_split=0.1, test_split=0.1, seed=42):
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)
        assert len(data_paths) == 1, "Only 1 path accepted for Bone Dataset"

        ds = CelebA(root=data_paths[0], split='all', download=False, transform=transform)
        self.path = f"{data_paths[0]}/{ds.base_folder}"
        ATTR = 'Blond_Hair'
        ATTR2 = 'Wearing_Necktie'
        attr_id = np.where(np.array(ds.attr_names) == ATTR)[0][0]
        attr_id2 = np.where(np.array(ds.attr_names) == ATTR2)[0][0]

        log.info("Using subset")
        NTH = 10
        filter_indices = np.zeros(len(ds.attr))
        filter_indices[::NTH] = 1
        log.info(f"Chosen attribute {ATTR} with id {attr_id}.")
        labels = ds.attr[:, attr_id]
        labels2 = ds.attr[:, attr_id2]
        both = np.where(np.logical_and(labels, labels2) == 1)[0]
        # print(both)
        filter_indices[both] = 1

        both_names = np.array(ds.filename)[both]
        # print("', '".join(both_names) + "\n")

        labels = ds.attr[:, attr_id][filter_indices == 1]
        labels2 = ds.attr[:, attr_id2][filter_indices == 1]
        both = np.where(np.logical_and(labels, labels2) == 1)[0]
        # print(both)

        labels_not_blonde = 1 * (ds.attr[:, attr_id][filter_indices == 1] == 0)
        labels_collar = ds.attr[:, attr_id2][filter_indices == 1]
        not_blonde_collar = np.where(np.logical_and(labels_not_blonde, labels_collar) == 1)[0]

        self.metadata = pd.DataFrame(
            {'image_id': np.array(ds.filename)[filter_indices == 1], 'targets': labels})

        pd_attr = pd.DataFrame(ds.attr[filter_indices == 1], columns=ds.attr_names[:-1])    # type: ignore
        pd_attr = pd_attr.reset_index(drop=True)
        pd_attr.columns = pd_attr.columns.astype(str)
        self.attributes = pd_attr.copy()
        self.sample_ids_by_concept = {
            attr: np.where(self.attributes[attr].values == 1)[0]
            for attr in self.attributes.columns
        }
        self.metadata = pd.concat([self.metadata.reset_index(drop=True), self.attributes], axis=1)
    
        self.normalize_fn = FN_NORMALIZE_CELEBA
        self.classes = [0, 1]
        self.class_names = [f'Non-{ATTR}', ATTR]
        self.num_classes = len(self.classes)

        self.mean = torch.Tensor(NORM_PARAMS_CELEBA[0])
        self.var = torch.Tensor(NORM_PARAMS_CELEBA[1])

        self.weights = self.compute_weights(np.array([len(labels) - labels.sum(), labels.sum()]))

        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(
            val_split=val_split,
            test_split=test_split,
            seed=seed
        )

        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()
        # transfer all artifacts to test set
        artifacts_in_train = [x for x in both if x in self.idxs_train]
        artifacts_to_keep = artifacts_in_train[::NTH]
        artifacts_to_remove = [x for x in artifacts_in_train if x not in artifacts_to_keep]
        self.idxs_train = np.array([x for x in self.idxs_train if x not in artifacts_to_remove])
        # select half of the artifacts to be in val and half in test
        artifacts_to_remove = np.array(artifacts_to_remove)
        np.random.default_rng(42).shuffle(artifacts_to_remove)
        self.idxs_val = np.concatenate([self.idxs_val, artifacts_to_remove[::2]])
        self.idxs_test = np.concatenate([self.idxs_test, artifacts_to_remove[1::2]])
        self.idxs_test.sort()
        self.idxs_val.sort()

    
        # log.info(f"Artifacts in train: {len([x for x in both if x in self.idxs_train])}")
        # log.info(f"Artifacts in test: {len([x for x in both if x in self.idxs_test])}")
        # log.info(f"Artifacts in val: {len([x for x in both if x in self.idxs_val])}")
        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

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
            image = self.augmentation(image)    # type: ignore

        return image.float(), target    # type: ignore

    def get_sample_name(self, i):
        return self.metadata.iloc[i]['image_id']

    def get_target(self, i):
        target = torch.tensor(self.metadata.iloc[i]["targets"])
        return target
    
    def get_num_classes(self):
        return len(self.classes)

    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset
    
    # Override
    def do_train_val_test_split(self, val_split=.1, test_split=.1, seed=0):
        rng = np.random.default_rng(seed=seed)
        idxs_all = np.arange(len(self))
        idxs_val = np.array(sorted(rng.choice(idxs_all, size=int(np.round(len(idxs_all) * val_split)), replace=False)))
        idxs_left = np.array(list(set(idxs_all) - set(idxs_val)))
        idxs_test = np.array(
            sorted(rng.choice(idxs_left, size=int(np.round(len(idxs_all) * test_split)), replace=False)))
        idxs_train = np.array(sorted(list(set(idxs_left) - set(idxs_test))))

        return idxs_train, idxs_val, idxs_test


if __name__ == "__main__":
    import torchvision
    data_paths = ["/Users/erogullari/datasets/"]
    ds = get_celeba_subset(data_paths, normalize_data=True, image_size=224)
    for i in range(10):
        img, _ = ds[i]
        torchvision.utils.save_image(ds.reverse_normalization(img).float() / 255.0, f"DELETE/celeba_subset_decs/dec{i}.png")
