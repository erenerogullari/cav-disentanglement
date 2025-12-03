import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image

class CelebAHQWithEncodings(Dataset):


    def __init__(
            self, 
            path_data: str, 
            path_encodings: str, 
            n_samples: int):
        
        super().__init__()
        self.path_data = Path(path_data)
        self.path_encodings = Path(path_encodings)

        self.get_paths()
        self.get_encs()
        self.length = min(len(self.paths), len(self.xs), n_samples)

    def get_paths(self):
        paths = list(self.path_data.rglob("*.png"))
        paths.sort(key=lambda x: int(x.parts[-2]))
        self.paths = paths


    def get_encs(self):
        # load all files
        data = [torch.load(e) for e in self.path_encodings.glob("*.pt")]
        idx = []
        xs = []
        ys = []
        id_to_label = None
        
        # collect elements in random order
        for d in data:
            # save indices to order the elements later
            idx.append(list(d.keys()))
            for v in d.values():
                xs.append(v["enc"])
                ys.append(list(v["labels"].values()))
                if id_to_label is None:
                    id_to_label = list(v["labels"].keys())
        
        # stack all elements
        self.xs = torch.stack(xs)
        self.ys = torch.Tensor(ys)

        # idx indicates the original index of each row
        idx = torch.LongTensor(sum(idx, [])).to(self.xs.device).unsqueeze(1)
        self.idx_ = idx # TODO: remove after debugging

        # set mapping for image indices
        mapping = idx.cpu()
        self.map_to_img_idx = lambda x: mapping[x]

        # reorder rows so that correct order is restored
        # self.xs.scatter_(0, idx.repeat(1, self.xs.shape[1]), xs)
        # self.ys.scatter_(0, idx.repeat(1, self.ys.shape[1]).cpu(), ys)
        
        # get mapping from label names to ids
        self.id_to_label = id_to_label
        self.label_to_id = {k: i for i, k in enumerate(id_to_label)}

        # extract mean and std of latents
        self.means = self.xs.mean(dim = 0)
        self.stds = self.xs.std(dim = 0)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        img_idx = self.map_to_img_idx(idx).item()
        img = read_image(str(self.paths[img_idx])) / 255
        enc = self.xs[idx]
        label = self.ys[idx]
        return img, enc, label, img_idx
    
    
    ###### CUSTOM ######
    def get_subset_by_idxs(self, idxs):
        """
        Returns a new dataset object containing only the samples at the specified indices.
        """
        subset = CelebAHQWithEncodings(self.path_data, self.path_encodings, len(idxs))
        
        # Update paths, xs, ys
        subset.paths = [self.paths[self.map_to_img_idx(i).item()] for i in idxs]
        subset.xs = self.xs[idxs]
        subset.ys = self.ys[idxs]
        
        # Update the map_to_img_idx for the subset
        subset.map_to_img_idx = lambda x: torch.tensor(idxs[x])
        
        # Update length and metadata
        subset.length = len(idxs)
        subset.id_to_label = self.id_to_label
        subset.label_to_id = self.label_to_id
        subset.means = subset.xs.mean(dim=0)
        subset.stds = subset.xs.std(dim=0)
        
        return subset