import copy
import numpy as np
import torch
import glob
import pandas as pd
import json
import torchvision.transforms as T
from datasets.base_dataset import BaseDataset
from PIL import Image
import os
import glob

PART_IDS = ["beak_model", "beak_color", "eye_model", "foot_model", 
            "tail_model", "tail_color", "wing_model", "wing_color"]


funnybirds_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)
])

def get_funnybirds(data_paths,
                        normalize_data=True,
                        image_size=32,
                        subfolder_name_extension="",
                        **kwargs):
    fns_transform = [T.Resize(image_size, interpolation=T.functional.InterpolationMode.BICUBIC), 
                     T.ToTensor()]
    if normalize_data:
        mean = torch.tensor([.5, .5, .5])
        std = torch.tensor([.5, .5, .5])
        fns_transform.append(T.Normalize(mean, std))

    transform = T.Compose(fns_transform)

    return FunnyBirds(data_paths, subfolder_name_extension, transform=transform, augmentation=funnybirds_augmentation)
    


class FunnyBirds(BaseDataset):

    def __init__(self, data_paths, subfolder_name_extension="", transform=None, augmentation=None):
        super().__init__(data_paths, transform, augmentation, None)
        self.metadata = self.load_metadata(f"{data_paths[0]}/train{subfolder_name_extension}", f"{data_paths[0]}/dataset_train{subfolder_name_extension}.json")        
        self.idxs_train, self.idxs_val, _ = self.do_train_val_test_split(.1, 0)
        self.metadata_test = self.load_metadata(f"{data_paths[0]}/test{subfolder_name_extension}", f"{data_paths[0]}/dataset_test{subfolder_name_extension}.json")
        self.idxs_test = np.arange(len(self), len(self) + len(self.metadata_test))
        self.metadata = pd.concat([self.metadata, self.metadata_test]).reset_index()

        self.mean = torch.Tensor((0.5, 0.5, 0.5))
        self.var = torch.Tensor((0.5, 0.5, 0.5))
        self.normalize_fn = T.Normalize(self.mean, self.var)

        self.class_names = [i for i in range(len(glob.glob(f"{data_paths[0]}/train{subfolder_name_extension}/*")))]
        self.labels = self.metadata["label"].array
        dist = np.array([(np.array(self.labels) == cl).sum() for cl in self.class_names])
        self.weights = self.compute_weights(dist)

        # Load classes and parts
        with open(data_paths[0] + '/classes.json', 'r') as file:
            self.classes = json.load(file)

        with open(data_paths[0] + '/parts.json', 'r') as file:
            self.parts = json.load(file)
    
    def load_metadata(self, path, path_part_data):
        samples = []
        classes = sorted([int(p.split("/")[-1]) for p in glob.glob(f"{path}/*")])
        for cl in classes:
            samples += [(os.path.basename(p)[:-4], p, cl) for p in sorted(glob.glob(f"{path}/{cl}/*"))]

        df_metadata = pd.DataFrame(samples, columns=["sample_id", "path", "label"]).sort_values("sample_id")
        
        with open(path_part_data) as file:
            part_info_json = json.load(file)

        part_data = []
        for obj in part_info_json:
            part_data.append([obj[part_id] for part_id in PART_IDS])
        df_part_data = pd.DataFrame(part_data, columns=PART_IDS)
        
        return pd.concat((df_metadata, df_part_data), axis=1)

    def __len__(self):
        return len(self.metadata)

    def get_target(self, i):
        return self.metadata["label"].array[i]
    
    def __getitem__(self, i):
        row = self.metadata.iloc[i]
        y = row["label"]
        x = Image.open(row['path']).convert("RGB")
        if self.transform:
            x = self.transform(x)
        if self.augmentation and self.do_augmentation:
            x = self.augmentation(x)
        return x, y
    
    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs]
        return subset
    
    def reverse_normalization(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone() + 0
        mean = self.mean.to(data)
        var = self.var.to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255).type(torch.int16)
    
    def get_parts(self):

        pass