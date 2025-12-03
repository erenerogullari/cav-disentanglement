import glob
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA

NORMALIZATION = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class CelebADataset(Dataset):
    def __init__(self, path_data, train=True, normalize_data=True, image_size=178, exclude=None, **kwargs):
        super().__init__()

        fns_transform = [
            T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
            # T.CenterCrop(image_size),
            T.RandomHorizontalFlip() if train else torch.nn.Identity(),
            # T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5) if train else torch.nn.Identity(),
            # T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25) if train else torch.nn.Identity(),
            # T.RandomApply(transforms=[T.Pad(10, fill=-(46.9 / 255.) / (22.6 / 255.)), T.Resize(image_size)], p=.25) if train else torch.nn.Identity(),
            T.ToTensor()
        ]

        if normalize_data:
            fns_transform.append(NORMALIZATION)

        self.transform = T.Compose(fns_transform)

        ds = CelebA(root=path_data, split='train' if train else 'test', download=False, transform=self.transform)
        self.path = f"{path_data}/{ds.base_folder}"

        if exclude and train:
            ATTR = exclude
            attr_id = np.where(np.array(ds.attr_names) == ATTR)[0][0]
            filtering = ds.attr[:, attr_id] == 0
            ds.filename = np.array(ds.filename)[filtering]
            ds.attr = np.array(ds.attr)[filtering]

        NTH = 10
        if train:
            print("Using subset")
            filtering = torch.ones(len(ds.attr))
            filtering[::NTH] = 0
            necktie = ds.attr[:, np.where((np.array(ds.attr_names) == 'Wearing_Necktie'))[0][0]]
            blonde = ds.attr[:, np.where((np.array(ds.attr_names) == 'Blond_Hair'))[0][0]]
            artifacts_lost = filtering * necktie * blonde

            filtering = torch.ones(len(ds.attr)) - artifacts_lost

            ds.attr = np.array(ds.attr)[filtering == 1]
            ds.filename = np.array(ds.filename)[filtering == 1]
            print(f"Using subset of {len(ds.attr)} samples.")
        else:
            # transfer rejected train artifacts to test set
            ds_train = CelebA(root=path_data, split='train', download=False, transform=self.transform)

            filtering = torch.ones(len(ds_train.attr))
            filtering[::NTH] = 0
            necktie = ds_train.attr[:, np.where((np.array(ds_train.attr_names) == 'Wearing_Necktie'))[0][0]]
            blonde = ds_train.attr[:, np.where((np.array(ds_train.attr_names) == 'Blond_Hair'))[0][0]]
            artifacts_transfer = np.where((filtering * necktie * blonde) == 1)[0].flatten()
            print(artifacts_transfer)
            print(f"Transferring {len(artifacts_transfer)} artifacts to test set.")
            ds.attr = np.concatenate([ds.attr, np.array(ds_train.attr)[artifacts_transfer]])
            ds.filename = np.concatenate([ds.filename, np.array(ds_train.filename)[artifacts_transfer]])

        self.attr = ds.attr
        self.attr_names = ds.attr_names

        ATTR = 'Blond_Hair'
        ATTR2 = 'Wearing_Necktie'
        attr_id = np.where(np.array(ds.attr_names) == ATTR)[0][0]
        attr_id2 = np.where(np.array(ds.attr_names) == ATTR2)[0][0]
        #
        print(f"Chosen attribute {ATTR} with id {attr_id}.")
        labels = ds.attr[:, attr_id]
        labels2 = ds.attr[:, attr_id2]
        both = np.where(np.logical_and(labels, labels2) == 1)[0]
        print(both)

        self.subsets = {
            'blonde_necktie': np.where(np.logical_and(1 * (ds.attr[:, attr_id] == 1), ds.attr[:, attr_id2]) == 1)[0],
            'not_blonde_necktie': np.where(np.logical_and(1 * (ds.attr[:, attr_id] == 0), ds.attr[:, attr_id2]) == 1)[
                0],
            'not_necktie': np.where(ds.attr[:, attr_id2] == 0)[0],
            'necktie': np.where(ds.attr[:, attr_id2] == 1)[0],
            'all': np.arange(len(ds.attr))
        }

        print(f"Chosen attribute {ATTR} with id {attr_id}.")
        self.metadata = pd.DataFrame(
            {'image_id': np.array(ds.filename), 'targets': labels})

        self.normalize_fn = NORMALIZATION

        self.classes = [f'Non-{ATTR}', ATTR]
        self.class_names = self.classes

        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.var = torch.Tensor([0.5, 0.5, 0.5])

        self.weights = torch.tensor(1 / np.array([len(labels) - labels.sum(), labels.sum()])).float()
        self.weights = self.weights / self.weights.sum()
        print(f"Weights: {self.weights}")

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")

        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        if self.transform:
            image = self.transform(image)

        return image.float(), target

    def get_sample_name(self, i):
        return self.metadata.iloc[i]['image_id']

    def get_target(self, i):
        target = torch.tensor(self.metadata.iloc[i]["targets"])
        return target


class CelebACleanDataset(CelebADataset):
    def __init__(self, path_data, train=True, normalize_data=True, image_size=178, **kwargs):
        super().__init__(path_data, train, normalize_data, image_size, exclude='Wearing_Necktie')


class CelebAModifiedDataset(CelebADataset):
    def __init__(self, path_data, train=True, normalize_data=True, image_size=178, **kwargs):
        super().__init__(path_data, train, normalize_data, image_size)

        PATH = "results/celeba_diffae_203k_diffae_log_reg/decode"
        modified = glob.glob(f"{PATH}/*")
        modified = [x.split("/")[-1] for x in modified]
        self.modified_paths = [f"{PATH}/{x}/img.png" for x in modified]

        fns_transform = [
            T.Resize((image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip() if train else torch.nn.Identity(),
            # T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5) if train else torch.nn.Identity(),
            T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)],
                          p=.25) if train else torch.nn.Identity(),
            # T.RandomApply(transforms=[T.Pad(10, fill=-(46.9 / 255.) / (22.6 / 255.)), T.Resize(image_size)], p=.25) if train else torch.nn.Identity(),
            T.ToTensor()
        ]

        if normalize_data:
            fns_transform.append(NORMALIZATION)

        self.transform_manip = T.Compose(fns_transform)

        in_modified = [f"{PATH}/{int(x.replace('.jpg', '')) - 1}/img.png" if str(
            int(x.replace(".jpg", "")) - 1) in modified else "" for x in self.metadata['image_id']]
        print(f"Modified images: {sum([x != '' for x in in_modified])} of {len(in_modified)}")

        self.metadata['image_id_pair'] = in_modified

        to_exclude = ['Wearing_Necktie']
        if False:
            for attr in to_exclude:
                attr_id = np.where(np.array(self.attr_names) == attr)[0][
                    0]  # np.where(np.array(ds.attr_names) == ATTR)[0][0]
                filtering = (self.attr[:, attr_id] == 1) * 1.0
                assert len(filtering) == len(self.metadata)
                has_no_pair = (self.metadata['image_id_pair'] == '') * 1.0

                # for faster testing every second sample
                every_second = np.array([x % 2 == 0 for x in range(len(self.metadata))])
                filtering = ((filtering + every_second * 1.0) >= 1.0) * 1.0

                self.metadata = self.metadata[(filtering * has_no_pair) == 0]

            labels = self.metadata['targets']
            self.weights = torch.tensor(1 / np.array([len(labels) - labels.sum(), labels.sum()])).float()
            self.weights = self.weights / self.weights.sum()
        print(f"Updated Weights: {self.weights}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")

        img_name_pair = self.metadata['image_id_pair'].iloc[idx]
        if img_name_pair:
            image_pair = Image.open(img_name_pair).convert("RGB")
        else:
            image_pair = None

        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        transform = self.transform_manip if img_name_pair else self.transform
        if self.transform:
            # apply SAME transform to original and modified sample
            seed = torch.randint(0, 100000, (1,)).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            image = transform(image).float()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            image_pair = transform(image_pair) if image_pair else torch.zeros_like(image)

        return image, target, image_pair.float()


class CelebADataset_RRR(CelebADataset):
    def __init__(self, path_data, train=True, normalize_data=True, image_size=178, artifact="necktie", **kwargs):
        super().__init__(path_data, train, normalize_data, image_size)

        self.hm_path = f"dataset/localized_artifacts/CelebADataset/{artifact}"
        self.hm_path += f"/train" if train else f"/test"
        artifact_paths = glob.glob(f"{self.hm_path}/*")
        artifact_sample_ids = np.array([int(x.split("/")[-1].split(".")[0]) for x in artifact_paths])
        self.artifact_ids = artifact_sample_ids
        self.hms = ["" for _ in range(len(self.metadata))]
        for i, j in enumerate(artifact_sample_ids):
            # print(i, j, len(self.hms))
            path = artifact_paths[i]
            if self.hms[j]:
                self.hms[j] += f",{path}"
            else:
                self.hms[j] += f"{path}"

        self.metadata["hms"] = self.hms
        print(self.hms[:10])
        fns_transform = [
            T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
            T.ToTensor()
        ]

        if normalize_data:
            fns_transform.append(NORMALIZATION)

        self.transform_only_resize = T.Compose(fns_transform)

    def __getitem__(self, i):

        if self.metadata["hms"].loc[i]:
            heatmaps = torch.stack(
                [torch.tensor(np.asarray(Image.open(hm))) for hm in self.metadata["hms"].loc[i].split(",")]).clamp(
                min=0).sum(0).float()
            self.transform_ = self.transform
            self.transform = self.transform_only_resize
            image, target = super().__getitem__(i)
            self.transform = self.transform_
        else:
            image, target = super().__getitem__(i)
            heatmaps = torch.zeros_like(image[0])

        return image, target, heatmaps


if __name__ == "__main__":
    ds = CelebADataset(["datasets"], train=True, transform=None, augmentation=None)
