import torch
from torch.utils.data import TensorDataset
from crp.attribution import CondAttribution
from crp.image import imgify
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
# from zennit.image import 
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from utils.model import truncate_and_extend_vgg16, truncate_and_extend_resnet18, FlattenAndMaxPool
from utils.cav import compute_all_cavs
from models.resnet import get_resnet18
from models.vgg import get_vgg16
from datasets import get_dataset


# Helper fn to compute relevances
def get_heatmaps(cavs, x, model, layer, device, model_type):
    if model_type == "vgg16":
        model_cav = truncate_and_extend_vgg16(model, layer, FlattenAndMaxPool(), cavs).to(device)
    elif model_type == "resnet18":
        model_cav = truncate_and_extend_resnet18(model, FlattenAndMaxPool(), cavs).to(device)
    else:
        raise ValueError("Unknown model type.")
    attribution = CondAttribution(model_cav)
    x = x.detach().to(device)
    x.requires_grad = True
    conditions = [{'y': 0}]
    composite = EpsilonPlusFlat()
    heatmaps, _, _, _ = attribution(x, conditions, composite)
    return heatmaps

# Helper fn to conormalize heatmaps
def conormalize_heatmaps(heatmaps, channel_avg=False):
    # Average over channels
    for i, hm in enumerate(heatmaps):
        if channel_avg:
            hm = hm.mean(dim=0)  # Shape: (H, W)
        heatmaps[i] = hm.cpu().numpy()

    # Compute global maximum absolute value for individual heatmaps
    all_heatmaps_array = np.stack(heatmaps)  # Shape: (N, H, W)
    global_abs_max_heatmaps = np.abs(all_heatmaps_array).max() + 1e-10  # Add epsilon to avoid division by zero

    # Normalize all heatmaps
    for i, hm in enumerate(heatmaps):
         heatmaps[i] /= global_abs_max_heatmaps

    return heatmaps

# Helper fn to visualize heatmaps
def visualize_heatmaps(image_tensor, heatmaps, channel_avg=False, conormalize=True, subplots_size=(1, 3), suptitle=None, titles=['Original Image', 'Before', 'After'], dot_products=None, fontsize=14):
    # Prepare all variables
    image = imgify(image_tensor)
    if conormalize:
        heatmaps = conormalize_heatmaps(heatmaps, channel_avg=channel_avg)
    
    # Check if subplot size fits
    n_rows, n_cols = subplots_size
    if len(heatmaps) != n_rows * (n_cols - 1):
        assert ValueError(f"The number of heatmaps ({len(heatmaps)}) doesn't match the subplot size {subplots_size}!")
    if len(titles) != n_cols:
        assert ValueError(f"The number of titles ({len(titles)}) doesn't match the number of columns in the subplot {(n_cols)}!") 
    if dot_products is not None and len(dot_products) != n_rows * (n_cols - 1):
        assert ValueError(f"The number of dot products ({len(dot_products)}) doesn't match the subplot size {subplots_size}!")

    # Define the color maps for heatmaps and intersection
    # cmap = plt.cm.bwr
    cmap = 'bwr'

    # Plots
    alpha = None
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 4 * n_rows))

    # Obtain a unified indexing for 1D or 2D axs
    axs = axs.reshape(n_rows, n_cols) if n_rows > 1 else [axs]

    # Loop through rows
    heatmap_idx = 0
    for i in range(n_rows):
        # First column: display original image
        axs[i][0].imshow(image)
        # Add row labels "before" and "after" for two rows
        if n_rows == 2:
            label = 'Before' if i == 0 else 'After'
            axs[i][-1].text(
                1.05, 0.5, label, transform=axs[i][-1].transAxes, 
                fontsize=fontsize, ha='left', va='center', rotation=90
            )
        axs[i][0].axis('off')  # Turn off axis for original image
        
        # Remaining columns: display heatmaps without background image
        for j in range(1, n_cols):
            ax = axs[i][j]
            heatmap_img = imgify(heatmaps[heatmap_idx], level=2, cmap=cmap, vmin=-1, vmax=1)
            ax.imshow(heatmap_img, alpha=alpha)
            ax.axis('off')  # Turn off axes for heatmap columns

            # Display dot product if available
            if dot_products is not None:
                dot_product_text = f"Dot Product = {dot_products[heatmap_idx]:.2f}"
                ax.text(
                    0.95, 0.05, dot_product_text, transform=ax.transAxes, 
                    fontsize=fontsize-2, ha='right', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                )
            heatmap_idx += 1

    # Titles
    for i in range(n_cols):
        axs[0][i].set_title(titles[i], fontsize=fontsize)
    
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.close(fig)  # Prevent display
    return fig


def generate_heatmaps_celeba(
        model_name,
        layer,
        cavs_name,
        model_path,
        concept_examples,
        entangled_examples,
        n_samples_each=2,
        n_samples_entangled=3,
        random_seed=42,
        device=None
    ):
    # Set defaults
    save_folder = f'media/{cavs_name}/heatmaps'
    layer = layer or ("features.28" if model_name == "vgg16" else "last_conv")
    # cavs_path = cavs_path or f"checkpoints/cavs_{model_name}.pt"

    # Set random seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Dataset initialization
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Dataset
    img_size = 224
    dataset_name = "celeba"
    dataset_dirs = ["/Users/erogullari/datasets/"]
    dataset = get_dataset(dataset_name)(data_paths=dataset_dirs, normalize_data=True, image_size=img_size)
    x_latent_all = torch.load(f"variables/{dataset_name}_latent_{model_name}.pt", map_location=device)
    labels = torch.tensor(dataset.attributes.to_numpy(), dtype=torch.float32).to(device)
    tensor_ds = TensorDataset(x_latent_all, labels)
    concepts = list(dataset.sample_ids_by_concept.keys())

    # Model initialization
    get_model = get_vgg16 if model_name == 'vgg16' else get_resnet18
    model = get_model(model_path, n_class=2)
    cavs_original = compute_all_cavs(x_latent_all.float(), labels.float()).to(device)
    cavs = torch.load(f'checkpoints/{cavs_name}.pt', map_location=device)
    if model_name == 'vgg16':
        model_cav_original = truncate_and_extend_vgg16(model, layer, FlattenAndMaxPool(), cavs_original).to(device)
        model_cav = truncate_and_extend_vgg16(model, layer, FlattenAndMaxPool(), cavs).to(device)
        canonizer = VGGCanonizer()
    elif model_name == 'resnet18':
        model_cav_original = truncate_and_extend_resnet18(model, FlattenAndMaxPool(), cavs_original).to(device)
        model_cav = truncate_and_extend_resnet18(model, FlattenAndMaxPool(), cavs).to(device)
        canonizer = ResNetCanonizer()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    composite = EpsilonPlusFlat(canonizers=[canonizer])
    attribution_original = CondAttribution(model_cav_original)
    attribution = CondAttribution(model_cav)

    # Prepare output folder
    os.makedirs(save_folder, exist_ok=True)

    # Generate single concept heatmaps
    print("Extracting heatmaps for single concepts...")
    for concept_id in concept_examples:
        samples = list(dataset.sample_ids_by_concept[concepts[concept_id]])
        sample_ids = random.sample(samples, n_samples_each)
        samples = [dataset[i][0].unsqueeze(0) for i in sample_ids]
        samples = torch.vstack(samples).to(device)
        samples.requires_grad_()
        conditions = [{"y": [concept_id]}]

        attr = attribution(samples, conditions, composite, record_layer=[layer])
        attr_original = attribution_original(samples, conditions, composite, record_layer=[layer])
        dot_products_before = x_latent_all[sample_ids] @ cavs_original[concept_id]
        dot_products_after = x_latent_all[sample_ids] @ cavs[concept_id]

        for i, sample in enumerate(samples):
            hms = [attr_original.heatmap[i], attr.heatmap[i]]
            dps = [dot_products_before[i], dot_products_after[i]]
            fig = visualize_heatmaps(sample, hms, suptitle=f"Concept: {concepts[concept_id]}", channel_avg=False, dot_products=dps)
            fig.savefig(f"{save_folder}/concept_{concepts[concept_id].replace(' ', '')}_heatmap{i}.png", bbox_inches="tight")

    print(f"Done. Single concept heatmaps saved in {save_folder}.")

    # Generate entangled concept heatmaps
    print("Extracting heatmaps for entangled concepts...")
    for pair in entangled_examples:
        concept1, concept2 = concepts[pair[0]], concepts[pair[1]]
        samples1 = dataset.sample_ids_by_concept[concept1]
        samples2 = dataset.sample_ids_by_concept[concept2]
        samples = list(set(samples1).intersection(set(samples2)))
        sample_ids = random.sample(samples, n_samples_entangled)
        samples = [dataset[i][0].unsqueeze(0) for i in sample_ids]
        samples = torch.vstack(samples).to(device)
        samples.requires_grad_()
        conditions = [{"y": [pair[0]]}, {"y": [pair[1]]}]

        attr = attribution(samples, conditions, composite, record_layer=[layer])
        attr_original = attribution_original(samples, conditions, composite, record_layer=[layer])
        dps_before = x_latent_all[sample_ids] @ cavs_original[list(pair)].T
        dps_after = x_latent_all[sample_ids] @ cavs[list(pair)].T
        hms_before = attr_original.heatmap.reshape((n_samples_entangled, 2, *attr.heatmap.shape[-2:])).permute(1, 0, 2, 3)
        hms_after = attr.heatmap.reshape((n_samples_entangled, 2, *attr.heatmap.shape[-2:])).permute(1, 0, 2, 3)

        for i, sample in enumerate(samples):
            hms = [hms_before[0][i], hms_before[1][i], hms_after[0][i], hms_after[1][i]]
            dps = [dps_before[i, 0], dps_before[i, 1], dps_after[i, 0], dps_after[i, 1]]
            fig = visualize_heatmaps(sample, hms, suptitle=f"Entangled Concepts: {concept1} & {concept2}", channel_avg=False, dot_products=dps)
            fig.savefig(f"{save_folder}/entangled_pair_{concept1.replace(' ', '')}-{concept2.replace(' ', '')}_{i}.png", bbox_inches="tight")

    print(f"Done. Entangled heatmaps saved in {save_folder}.")