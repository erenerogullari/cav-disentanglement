import torch
import logging
import wandb
import os
import numpy as np
import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
from models import get_fn_model_loader
from utils.helper import get_device, load_config
import matplotlib.pyplot as plt
from utils.localization import get_localizations
from torch.utils.data import DataLoader
from crp.attribution import CondAttribution
from models import  get_canonizer
from zennit.composites import EpsilonPlusFlat
from sklearn.metrics import jaccard_score
from utils.localization import binarize_heatmaps
import seaborn as sns
import pandas as pd
from crp.image import imgify

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        # default="config_files/training/celeba_attacked/local/vgg16_p_artifact0.3_factor100_sgd_lr0.001.yaml")
                        default="config_files/cav_disentanglement/celeba_attacked/local/vgg16_p_artifact0.4_factor5_optimsgd0.001_alpha1_beta100_lr0.0001.yaml")
    parser.add_argument('--plot_to_wandb', default=True, type=bool)
    parser.add_argument('--after_disentanglement', default=True, type=bool)
    parser.add_argument('--cav_type', default="best", type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    config = load_config(args.config_file)
    plot_concept_heatmaps(config, args.plot_to_wandb, after_disentanglement=args.after_disentanglement, cav_type=args.cav_type)


def plot_concept_heatmaps(config, plot_to_wandb, num_imgs=16, after_disentanglement=True, cav_type="best"):
    device = get_device()
    wandb_project_name = config.get('wandb_project_name', None)
    wandb_api_key = config.get('wandb_api_key', None)

    do_wandb_logging = wandb_project_name is not None
    # Initialize WandB
    if do_wandb_logging:
        assert wandb_api_key is not None, f"'wandb_api_key' required if 'wandb_project_name' is provided ({wandb_project_name})"
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(id=config['wandb_id'], project=wandb_project_name, 
                   config=config, resume=True)
        wandb.run.name = f"{config['config_name']}-{wandb.run.name}"
        logger.info(f"Initialized wand. Logging to {wandb_project_name} / {wandb.run.name}...")

    dataset = load_dataset(config, normalize_data=True, hm=True)
    model = get_fn_model_loader(config["model_name"])(n_class=len(dataset.classes), ckpt_path=config["ckpt_path"], device=device).to(device).eval()
    
    if after_disentanglement:
        model_name = config["model_name"]
        p_artifact = config["p_artifact"]
        factor = config["entanglement_factor"]
        optimizer = config["optimizer"]
        lr = config["base_lr"]
        base_path_cavs = f"{config['dir_precomputed_data']}/cavs/{config['dataset_name']}/{model_name}_p_artifact{p_artifact}_factor{factor}_{optimizer}_lr{lr}"
        path_cavs_baseline = f"{base_path_cavs}/cavs_baseline.pth"
        path_cavs_orthogonal = f"{base_path_cavs}/cavs_disentangled_{config['config_name']}_{cav_type}.pth"
        # path_cavs_orthogonal = f"notebooks/data/cavs_disentangled_ab.pt"
        # path_cavs_orthogonal = f"tmp/cavs.pt"
        cavs_all = {
            "Baseline": torch.load(path_cavs_baseline),
            "Orthogonal": torch.load(path_cavs_orthogonal)
        }
    else:
        base_path_cavs = f"{config['dir_precomputed_data']}/cavs/{config['dataset_name']}/{config['config_name']}"
        path_cavs_baseline = f"{base_path_cavs}/cavs_baseline.pth"
        cavs_all = {
            "Baseline": torch.load(path_cavs_baseline)
        }

    cnames_all = list(dataset.metadata.columns.values[2:])
    assert len(cavs_all["Baseline"]) == len(cnames_all), f"mismatch between cav tensor ({cavs.shape}) and concept names {len(cnames_all)}"
    concepts = ["box", "timestamp", "Blond_Hair"]

    sample_ids = np.intersect1d(np.intersect1d(dataset.art1_ids, 
                                               dataset.art2_ids), 
                                dataset.idxs_test)
    logger.info(f"Found {len(sample_ids)} interesting samples.")
    ds_art_test = dataset.get_subset_by_idxs(sample_ids)

    attribution = CondAttribution(model)
    canonizers = get_canonizer(config["model_name"])
    composite = EpsilonPlusFlat(canonizers)

    cav_localizations = {}
    for name, cavs in cavs_all.items():
        cavs_subset = {c: cavs[cnames_all.index(c)] for c in concepts}
        imgs, localizations, gts = compute_concept_relevances(attribution, ds_art_test, cavs_subset, composite, config, device)
        cav_localizations[name] = localizations

    savepath = f"{config['dir_outputs']}/concept_visualizations/{config['config_name']}_{cav_type}"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    create_plot(dataset, imgs[:min(len(imgs), num_imgs)], cav_localizations, gts["timestamp"], gts["box"], savepath)

    ## quantify
    results_quant = {}
    for cav_name, _ in cavs_all.items():
        for cname in ["timestamp", "box"]:
            loc = cav_localizations[cav_name][cname]
            concept_rel = ((loc * gts[cname]).sum((1,2)) / (loc.sum((1, 2)) + 1e-10))
            loc_concept_binary = binarize_heatmaps(loc, thresholding="otsu")
            jaccards = np.array([jaccard_score(loc_concept_binary[i].reshape(-1).numpy(),  gts[cname][i].reshape(-1).numpy()) for i in range(len(loc_concept_binary))])
            concept_rels = concept_rel
            concept_iou = jaccards
            results_quant[f"iou_{cname}_{cav_name}_{cav_type}"] = concept_iou.mean()
            results_quant[f"concept_rel_{cname}_{cav_name}_{cav_type}"] = concept_rels.mean()

    data_plot = pd.DataFrame(data=[
        ("Baseline", results_quant["concept_rel_timestamp_Baseline_best"].item()),
        ("Orthogonal", results_quant["concept_rel_timestamp_Orthogonal_best"].item()),
            ], columns=["CAV", "Concept Relevance"])
    savepath_quant = f"{config['dir_outputs']}/concept_visualizations/{config['config_name']}_{cav_type}_concept_relevance"
    vmax = 0.5 if results_quant["concept_rel_timestamp_Orthogonal_best"].item() > 0.44 else 0.45
    plot_concept_relevance(data_plot, vmax, savepath_quant)

    if plot_to_wandb:
        str_type = "-".join(list(cav_localizations.keys()))
        wandb.log({f"concept_heatmaps_{str_type}_{cav_type}": wandb.Image(f"{savepath}.jpg")})
        wandb.log(results_quant)

def plot_concept_relevance(data_plot, vmax, savename):
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 9, 'legend.fontsize': 9, 'axes.titlesize': 11})
    fig = plt.figure(figsize=(2.5, 3))
    # sns.barplot(data=data_plot, y="Concept Relevance")
    sns.barplot(x='CAV', y='Concept Relevance', hue="CAV", data=data_plot)
    ticks = [0.25, 0.3, 0.35, 0.4, 0.45]
    if vmax == .5:
        ticks.append(0.5)
    plt.ylim(0.25, vmax)
    plt.yticks(ticks)
    plt.show()
    [fig.savefig(f"{savename}.{ending}", bbox_inches="tight", dpi=500) for ending in ["png", "jpg", "pdf"]]
    plt.close()

def compute_concept_relevances(attribution, ds, cavs, composite, config, device, batch_size=8):
    localizations = {c: None for c in cavs.keys()}
    gts = {"timestamp": None, "box": None}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    imgs = None
    for i, (x, _, loc1, loc2) in enumerate(tqdm.tqdm(dl)):
        for cname, cav in cavs.items():
            attr, loc_cav = get_localizations(x.clone(), cav, attribution, composite, config, device)
            loc_cav = attr.heatmap.detach().cpu().clamp(min=0)
            localizations[cname] = loc_cav if localizations[cname] is None else torch.cat([localizations[cname], loc_cav])
        gts["timestamp"] = loc1 if gts["timestamp"] is None else torch.cat([gts["timestamp"], loc1])
        gts["box"] = loc2 if gts["box"] is None else torch.cat([gts["box"], loc2])
        imgs = x.detach().cpu() if imgs is None else torch.cat([imgs, x.detach().cpu()])
    return imgs, localizations, gts

def create_plot(ds, imgs, cav_localizations, gt_timestamp, gt_box, savepath):
    num_cavs = len(cav_localizations)

    nrows = len(imgs)
    ncols = 3 + 3 * num_cavs
    size = 1.7
    level = 2.0
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*size, nrows*size))
    
    for i in range(nrows):
        
        # input
        ax = axs[i][0]
        ax.imshow(ds.reverse_normalization(imgs[i]).permute((1, 2, 0)).int().numpy())
        axs[0][0].set_title("Input")

        all_maxs = [loc[i].max() for _, loc in cav_localizations["Baseline"].items()]

        for cav_idx, (cav_name, localizations) in enumerate(cav_localizations.items()):

        # hm timestamp
            cname = "timestamp"
            # all_maxs = [all_concept_hms[cname][i].max() for _, all_concept_hms in cav_localizations.items()]
            all_maxs = [all_concept_hms[cname][i].max() for n, all_concept_hms in cav_localizations.items() if n == cav_name]
            normalization_constant = torch.max(torch.tensor(all_maxs))
            c = 1 + cav_idx
            ax = axs[i][c]
            img_hm = imgify(localizations[cname][i] / normalization_constant, 
                            cmap="bwr", vmin=-1, vmax=1, level=level)
            ax.imshow(img_hm)
            axs[0][c].set_title(f"{cname}\n{cav_name}")

            cname = "box"
            # all_maxs = [all_concept_hms[cname][i].max() for _, all_concept_hms in cav_localizations.items()]
            all_maxs = [all_concept_hms[cname][i].max() for n, all_concept_hms in cav_localizations.items() if n == cav_name]
            normalization_constant = torch.max(torch.tensor(all_maxs))
            c = 1 + num_cavs + 1 + cav_idx
            ax = axs[i][c]
            img_hm = imgify(localizations[cname][i] / normalization_constant, 
                            cmap="bwr", vmin=-1, vmax=1, level=level)
            ax.imshow(img_hm)
            axs[0][c].set_title(f"{cname}\n{cav_name}")

        
            cname = "Blond_Hair"
            # all_maxs = [all_concept_hms[cname][i].max() for _, all_concept_hms in cav_localizations.items()]
            all_maxs = [all_concept_hms[cname][i].max() for n, all_concept_hms in cav_localizations.items() if n == cav_name]
            normalization_constant = torch.max(torch.tensor(all_maxs))
            c = 1 +  2 * (num_cavs + 1) + cav_idx
            ax = axs[i][c]
            img_hm = imgify(localizations[cname][i] / normalization_constant, 
                            cmap="bwr", vmin=-1, vmax=1, level=level)
            ax.imshow(img_hm)
            axs[0][c].set_title(f"{cname}\n{cav_name}")


        # gts
        c = 1 + num_cavs
        ax = axs[i][c]
        ax.imshow(gt_timestamp[i].numpy())
        axs[0][c].set_title(f"Ground Truth")

        # gt
        c = 1 + 2 * num_cavs + 1
        ax = axs[i][c]
        ax.imshow(gt_box[i].numpy())
        axs[0][c].set_title(f"Ground Truth")

    for _axs in axs:
        for ax in _axs:
            ax.set_xticks([])
            ax.set_yticks([])

    print(f"Storing at {savepath}")
    [fig.savefig(f"{savepath}.{ending}", bbox_inches="tight") for ending in ["jpg", "pdf"]]


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()