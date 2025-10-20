import os
import torch
from crp.attribution import CondAttribution
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import random
from utils.cav import compute_cavs
from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.model import truncate_and_extend, FlattenAndMaxPool
from DELETE.heatmaps_DELETE import visualize_heatmaps



entangled_pairs =[
    (9, 38),        # Blond Hair - Wearing Necktie
    (4, 31),        # Bald - Smiling
    (8, 38),        # Black Hair - Wearing Necktie
    (16, 9),        # Goatee - Blond Hair
    (16, 8),        # Goatee - Black Hair
]


@hydra.main(version_base=None, config_path="../configs/extract_heatmaps", config_name="config.yaml")
def main(config: DictConfig) -> None:

    device = config.attribution.device
    print(f"Using device: {device}")
    random.seed(config.attribution.random_seed)

    # Initialize dataset
    dataset = get_dataset(config.dataset.dataset)(data_paths=[config.dataset.path],
                                        normalize_data=True,
                                        image_size=config.dataset.img_size)

    # Initialize model and attribution
    model_path = f"checkpoints/checkpoint_{config.model.model}_{config.dataset.dataset}.pth"
    model = get_fn_model_loader(model_name=config.model.model)(n_class=config.model.n_class, ckpt_path=model_path).to(device)
    attribution = CondAttribution(model)
    canonizer = get_canonizer(config.model.model)
    composite = instantiate(config.attribution.composite)

    # Load latent variables to compute baseline (entangled) cavs
    x_latent_all = torch.load(f'variables/latents_{config.dataset.dataset}_{config.model.model}.pt', weights_only=True)
    labels = dataset.get_labels()
    concept_names = dataset.get_concept_names()

    # Compute and load cavs
    cavs_original = compute_cavs(x_latent_all.float(), labels.float())
    cavs_orthogonal = torch.load(f"{config.cavs.path}/cavs.pt", weights_only=True)

    # Truncate and extend to obtain attribution models
    model_cav_original = truncate_and_extend(config.model.model, model, config.model.layer, FlattenAndMaxPool(), cavs_original).to(device)
    model_cav_orthogonal = truncate_and_extend(config.model.model, model, config.model.layer, FlattenAndMaxPool(), cavs_orthogonal).to(device)
    attribution_original = CondAttribution(model_cav_original)
    attribution_orthogonal = CondAttribution(model_cav_orthogonal)

    # Extract heatmaps 
    save_folder = f"{config.cavs.path}/media/heatmaps"
    os.makedirs(save_folder, exist_ok=True)


    for pair in entangled_pairs:
        n_samples = config.attribution.n_samples_each
        concept1 = concept_names[pair[0]]
        concept2 = concept_names[pair[1]]
        samples1 = dataset.sample_ids_by_concept[concept1]
        samples2 = dataset.sample_ids_by_concept[concept2]

        # Get the intersection
        samples = list(set(samples1).intersection(set(samples2)))
        sample_ids = random.sample(samples, n_samples)
        samples = [dataset[i][0].unsqueeze(0) for i in sample_ids]
        samples = torch.vstack(samples)         # Shape [n_samples_each, 3, 224, 224]
        samples.requires_grad_()

        # Obtain Heatmaps
        conditions = [{'y': [pair[0]]}, {'y': [pair[1]]}]
        attr = attribution_orthogonal(samples.to(device), conditions, composite, record_layer=[config.model.layer])
        attr_original = attribution_original(samples.to(device), conditions, composite, record_layer=[config.model.layer])
        # dps_before = x_latent_all[sample_ids] @ cavs_original[list(pair)].T               # Shape [n, 2]
        # dps_after = x_latent_all[sample_ids] @ cavs_orthogonal[list(pair)].T              # Shape [n, 2]
        hms_before = attr_original.heatmap.reshape((n_samples, 2, attr.heatmap.shape[-2], attr.heatmap.shape[-1])).permute(1,0,2,3)
        hms_after = attr.heatmap.reshape((n_samples, 2, attr.heatmap.shape[-2], attr.heatmap.shape[-1])).permute(1,0,2,3)

        for i, sample in enumerate(samples):
            hms = [hms_before[0][i], hms_before[1][i], hms_after[0][i], hms_after[1][i]]
            # dps = [dps_before[i, 0], dps_before[i, 1], dps_after[i, 0], dps_after[i, 1]]
            # suptitle = 'Entangled Concepts'
            dps = None
            suptitle = None
            # Visualize
            subtitles = ['Original Image' ,f'{concept1}', f'{concept2}']
            fig = visualize_heatmaps(sample.detach(), hms ,subplots_size=(2,3) ,suptitle=suptitle, channel_avg=False, titles=subtitles, dot_products=dps, fontsize=14)
            fig.savefig(f"{save_folder}/entangled_pair_{concept1.replace(' ', '')}-{concept2.replace(' ', '')}_{i}.pdf", format='pdf', bbox_inches='tight')

    print(f'Done. Heatmaps saved in {save_folder}.')


if __name__ == "__main__":
    main()