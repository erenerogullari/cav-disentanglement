import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CelebAHQ(Dataset):


    def __init__(self, path_data: str, path_labels: str, n_samples: int, name: str):
        super().__init__()
        self.path_data = Path(path_data)
        self.path_labels = Path(path_labels)
        self.name = name

        self.paths = self.get_paths()
        self.labels = self.get_labels()
        self.length = min(len(self.paths), n_samples)


    def get_paths(self):
        paths = list(self.path_data.rglob("*.png"))
        paths.sort(key=lambda x: int(x.parts[-2]))
        return paths


    def get_labels(self):
        df_labels = pd.read_csv(self.path_labels, index_col = 0)
        return df_labels


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        img = read_image(str(self.paths[idx])) / 255
        labels = self.labels.loc[idx].to_dict()
        return img, labels, idx
    
    def get_concept_names(self):
        return list(self.labels.columns)
    
    def do_train_val_test_split(self, val_split: float, test_split: float, seed: int = 0):
        n_total = len(self)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test

        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        return (
            train_indices,
            val_indices,
            test_indices,
        )
    
