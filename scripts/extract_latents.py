import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from crp.attribution import CondAttribution
from zennit.composites import EpsilonPlusFlat
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
import pickle


def get_features(batch, layer_name, attribution, canonizer, cav_mode, device):
    compost = EpsilonPlusFlat(canonizers=canonizer)
    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    attr = attribution(batch.to(device), dummy_cond, compost, record_layer=[layer_name])
    if cav_mode == "full":
        features = attr.activations[layer_name]
    else:
        acts = attr.activations[layer_name]
        features = acts.flatten(start_dim=2).max(2)[0]
    return features

@hydra.main(version_base=None, config_path="../configs/extract_latents", config_name="config.yaml")
def main(config: DictConfig) -> None:

    device = config.inference.device
    print(f"Using device: {device}")

    # Initialize dataset and necessary variables
    dataset = get_dataset(config.dataset.dataset)(data_paths=[config.dataset.path],
                                        normalize_data=True,
                                        image_size=config.dataset.img_size)
    print(f"Total number of samples: {len(dataset)}")
    labels = dataset.get_labels()
    concept_names = dataset.get_concept_names()
    dataloader = DataLoader(dataset, batch_size=config.inference.batch_size)

    # Initialize model and attribution
    model_path = f"checkpoints/checkpoint_{config.model.model}_{config.dataset.dataset}.pth"
    model = get_fn_model_loader(model_name=config.model.model)(n_class=config.model.n_class, ckpt_path=model_path).to(device)
    attribution = CondAttribution(model)
    canonizer = get_canonizer(config.model.model)

    # Obtain activations
    x_latent_all = None
    for batch in tqdm(dataloader):
        x = batch[0]
        x_latent = get_features(x, config.model.layer, attribution, canonizer, config.model.cav_mode, device=device)
        x_latent = x_latent.detach().cpu()
        x_latent_all = x_latent if x_latent_all is None else torch.cat((x_latent_all, x_latent))

    # Save the results
    latents_save_path = f"variables/latents_{config.dataset.dataset}_{config.model.model}.pt"
    torch.save(x_latent_all, latents_save_path)
    print(f"Latents saved to '{latents_save_path}'")

    labels_save_path = f"variables/labels_{config.dataset.dataset}.pt"
    concepts_save_path = f"variables/concept_names_{config.dataset.dataset}.pt"
    if not os.path.exists(labels_save_path):
        torch.save(labels, labels_save_path)
        print(f"Labels saved to '{labels_save_path}'")
    if not os.path.exists(concepts_save_path):
        with open(concepts_save_path, 'wb') as f:
            pickle.dump(concept_names, f)
        print(f"Concept names saved to '{concepts_save_path}'")
    

if __name__ == "__main__":
    main()