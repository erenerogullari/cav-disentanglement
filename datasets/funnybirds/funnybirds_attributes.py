import torch
import torchvision.transforms as T
from datasets.funnybirds.funnybirds import FunnyBirds, funnybirds_augmentation, PART_IDS
import numpy as np
import pandas as pd

def get_funnybirds_attributes(data_paths,
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

    return FunnyBirdsAttributes(data_paths, subfolder_name_extension, transform=transform, augmentation=funnybirds_augmentation)
    

class FunnyBirdsAttributes(FunnyBirds):

    def __init__(self, data_paths, subfolder_name_extension, transform=None, augmentation=None):
        super().__init__(data_paths, subfolder_name_extension, transform, augmentation)

        self.attributes = self.find_all_attributes()
        self.sample_ids_by_concept = self.build_sample_ids_by_concept()     # added this
        

    def find_all_attributes(self):
        parts = np.unique(np.array([name.split("_")[0] for name in self.metadata.columns[4:]]))
        all_attrs = np.unique(np.array([self.get_attr(r, p) for _, r in self.metadata.iterrows() for p in parts]))
        all_attrs_no_placeholder = np.array([attr for attr in all_attrs if "placeholder" not in attr])
        return all_attrs_no_placeholder

    def get_attr(self, r, p):
        attr = f"{p}::{r[p + '_model']}"
        attr += f"::{r[p + '_color']}" if p not in ("eye", "foot") else ""
        return attr
    
    def check_attr(self, r, attr):
        part = attr.split("::")[0]
        
        contains_attr = r[f"{part}_model"] == attr.split("::")[1]
        if part not in ("eye", "foot"):
            contains_attr = contains_attr and r[f"{part}_color"] == attr.split("::")[2]
        return contains_attr * 1
    
    def get_attribute_labels(self, i):
        row = self.metadata.iloc[i]
        return torch.tensor([self.check_attr(row, attr)for attr in self.attributes]).type(torch.long)

    def build_pos_neg_concept_indexes(self):
        ## Pos Samples
        concept_index_pos = {n: [] for i, n in enumerate(self.attributes)}
        for i in range(len(self)):
            for attr_idx, attr_label in enumerate(self[i][2]):
                if attr_label == 1:
                    concept_index_pos[self.attributes[attr_idx]].append(i)
        
        ## Neg Samples
        concept_index_neg = {n: [] for i, n in enumerate(self.attributes)}
        for c_key, idxs_pos in concept_index_pos.items():
            rng = np.random.default_rng(0)
            idxs_neg_all = list(set(np.arange(len(self))) - set(idxs_pos))
            replace = len(idxs_neg_all) < len(idxs_pos)
            idxs_neg = rng.choice(idxs_neg_all, len(idxs_pos), replace=replace)
            concept_index_neg[c_key] = idxs_neg

        return concept_index_pos, concept_index_neg
    
    def get_labels(self):
        return torch.tensor(self.get_binary_dataset().values, dtype=torch.long)
    
    def get_concept_names(self):
        return self.find_all_attributes()
    
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        attr = self.get_attribute_labels(i)
        return x, y, attr

    def get_binary_dataset(self):
        """
        Generate a pandas DataFrame combining metadata and binary attribute labels.
        Returns the DataFrame and displays the first few rows for inspection.
        """
        # Generate binary labels for all samples
        binary_labels = pd.DataFrame(
            [self.get_attribute_labels(i).numpy() for i in range(len(self))], 
            columns=self.attributes
        )

        return binary_labels
    
    def build_sample_ids_by_concept(self):
        """
        Builds a dictionary mapping each concept (attribute) to the list of sample IDs 
        where that concept is present.
        """
        sample_ids_by_concept = {attr: [] for attr in self.attributes}

        for i in range(len(self)):
            _, _, attr_labels = self[i]  # Get the attribute labels for the sample
            for attr_idx, label in enumerate(attr_labels):
                if label == 1:  # Sample belongs to the concept
                    sample_ids_by_concept[self.attributes[attr_idx]].append(i)
        
        return sample_ids_by_concept