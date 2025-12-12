import copy
import json
import pandas as pd
import torch
import logging
import wandb
import os
import numpy as np
import tqdm
import seaborn as sns
from argparse import ArgumentParser
from datasets import load_dataset
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.distance_metrics import cosine_similarities_batch
from utils.helper import get_device, load_config
import matplotlib.pyplot as plt
from utils.localization import get_localizations
from torch.utils.data import DataLoader
from crp.attribution import CondAttribution
from models import  get_canonizer
from zennit.composites import EpsilonPlusFlat
from sklearn.metrics import jaccard_score
from utils.localization import binarize_heatmaps
from experiments.mitigation_experiments.start_model_correction import start_model_correction
from experiments.evaluation.evaluate_by_subset_attacked import evaluate_by_subset_attacked

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        default="config_files/cav_disentanglement/celeba_attacked/local/vgg16_p_artifact0.4_factor5_optimsgd0.001_alpha1_beta100_lr0.0001.yaml")
    parser.add_argument('--plot_to_wandb', default=True, type=bool)
    parser.add_argument('--cav_type', default="best", type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    config = load_config(args.config_file)
    eval_pclarc(config, args.cav_type)


def eval_pclarc(config, cav_type):
    device = get_device()
    print(f"Using device {device}")

    dataset = load_dataset(config, normalize_data=True)
        
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

    savepath = f"{config['dir_outputs']}/cav_disentanglement_pclarc"
    os.makedirs(savepath, exist_ok=True)
    config['checkpoint_dir_corrected'] = config['dir_precomputed_data'] 
    config['lr'] = config['base_lr'] 
    config_vanilla = copy.deepcopy(config)
    config_vanilla["num_epochs"] = 0
    config_vanilla["artifact"] = "timestamp"
    config_vanilla["wandb_api_key"] = None
    config_vanilla["method"] = "Vanilla"
    config_pclarc = copy.deepcopy(config_vanilla)
    config_pclarc["method"] = "PClarc"
    start_model_correction(config_pclarc, num_gpu=1)
    start_model_correction(config_vanilla, num_gpu=1)

    ## Load CAVs
    model_name = config["model_name"]
    p_artifact = config["p_artifact"]
    factor = config["entanglement_factor"]
    optimizer = config["optimizer"]
    lr = config["base_lr"]
    base_path_cavs = f"{config['dir_precomputed_data']}/cavs/{config['dataset_name']}/{model_name}_p_artifact{p_artifact}_factor{factor}_{optimizer}_lr{lr}"
    # path_cavs_baseline = f"{base_path_cavs}/cavs_baseline.pt"
    # path_cavs_orthogonal = f"{base_path_cavs}/cavs_disentangled.pt"
    path_scavs_baseline = f"{base_path_cavs}/scavs_baseline.pt"
    path_scavs_orthogonal = f"{base_path_cavs}/scavs_disentangled.pt"
    path_lcavs_baseline = f"{base_path_cavs}/lcavs_baseline.pt"
    path_lcavs_orthogonal = f"{base_path_cavs}/lcavs_disentangled.pt"
    path_lcavs_orthogonalv2 = f"{base_path_cavs}/lcavs_disentangled_orth.pt"
        
    # cavs_baseline = torch.load(path_cavs_baseline, weights_only=True, map_location='cpu')
    # cavs_orthogonal = torch.load(path_cavs_orthogonal, weights_only=True, map_location='cpu')

    scavs_baseline = torch.load(path_scavs_baseline, weights_only=True, map_location='cpu')
    scavs_orthogonal = torch.load(path_scavs_orthogonal, weights_only=True, map_location='cpu')
    lcavs_baseline = torch.load(path_lcavs_baseline, weights_only=True, map_location='cpu')
    lcavs_orthogonal = torch.load(path_lcavs_orthogonal, weights_only=True, map_location='cpu')
    lcavs_orthogonalv2 = torch.load(path_lcavs_orthogonalv2, weights_only=True, map_location='cpu')

    all_cavs = {
        'SCAVs Baseline': scavs_baseline,
        'SCAVs Orthogonal': scavs_orthogonal,
        'LCAVs Baseline': lcavs_baseline,
        'LCAVs Orthogonal': lcavs_orthogonal,
        'LCAVs Orthogonal V2': lcavs_orthogonalv2
    }
    
    with open(f"{base_path_cavs}/concept_names.pth", "r") as f:
        concepts = json.load(f)

    concept_id = concepts.index("timestamp")

    ## Concept similarity stuff
    model = get_fn_model_loader(config["model_name"])(n_class=len(dataset.classes), ckpt_path=config["ckpt_path"], device=device).to(device).eval()
    model_pclarc = prepare_model_for_evaluation(copy.deepcopy(model), dataset, device, config_pclarc)

    model_pclarc_disentangled = copy.deepcopy(model_pclarc)
    model_pclarc_disentangled.cav = scavs_orthogonal[concept_id].unsqueeze(0)

    split = "test"
    activations_vanilla, _ = get_activations_ds(model, dataset, config, device, split)
    activations_pclarc_baseline, metadata_concepts = get_activations_ds(model_pclarc, dataset, config, device, split)
    activations_pclarc_orthogonal, metadata_concepts = get_activations_ds(model_pclarc_disentangled, dataset, config, device, split)

    # similarities_vanilla = {cname: cosine_similarities_batch(activations_vanilla[metadata_concepts[cname] == 1], cavs_baseline[concepts.index(cname)]) for cname in concepts}
    # similarities_pclarc_baseline = {cname: cosine_similarities_batch(activations_pclarc_baseline[metadata_concepts[cname] == 1], cavs_baseline[concepts.index(cname)]) for cname in concepts}
    # similarities_pclarc_orthogonal = {cname: cosine_similarities_batch(activations_pclarc_orthogonal[metadata_concepts[cname] == 1], cavs_orthogonal[concepts.index(cname)]) for cname in concepts}

    similarities_dict = {
        'Vanilla': {
            cname: cosine_similarities_batch(
                activations_vanilla[metadata_concepts[cname] == 1],
                scavs_baseline[concepts.index(cname)]
            ) for cname in concepts
    }}
    for name, cavs in all_cavs.items():
        similarities_dict[name] = {
            cname: cosine_similarities_batch(
                activations_pclarc_baseline[metadata_concepts[cname] == 1],
                cavs[concepts.index(cname)]
            ) for cname in concepts
        }

    data_similarities = []
    # for c, sims in similarities_vanilla.items():
    #     for v in sims:
    #         data_similarities.append({'concept': c, 'v': v.item(), 'model': 'Vanilla'})
    # for c, sims in similarities_pclarc_baseline.items():
    #     for v in sims:
    #         data_similarities.append({'concept': c, 'v': v.item(), 'model': 'Baseline CAV'})
    # for c, sims in similarities_pclarc_orthogonal.items():
    #     for v in sims:
    #         data_similarities.append({'concept': c, 'v': v.item(), 'model': 'Orthogonal CAV'})
    for model_name, sims_dict in similarities_dict.items():
        for c, sims in sims_dict.items():
            for v in sims:
                data_similarities.append({'concept': c, 'v': v.item(), 'model': model_name})

    df_similarities = pd.DataFrame(data_similarities)
    savename = f"{savepath}/concept_similarity_after_pclarc_{config['config_name']}_{cav_type}"
    plot_concept_similarities(df_similarities, savename)
    ##

    # accuracy_metrics_vanilla, cm_vanilla = evaluate_by_subset_attacked(config_vanilla, return_cm=True)
    # accuracy_metrics_pclarc_baseline, cm_pclarc_baseline = evaluate_by_subset_attacked(config_pclarc, 
    #                                                                                    cav=cavs_baseline[concept_id].unsqueeze(0),
    #                                                                                    return_cm=True)
    # accuracy_metrics_pclarc_orthogonal, cm_pclarc_orthogonal = evaluate_by_subset_attacked(config_pclarc, 
    #                                                                                         cav=cavs_orthogonal[concept_id].unsqueeze(0),
    #                                                                                         return_cm=True)

    # Evaluate Vanilla (no CAV) model
    metrics_vanilla, cm_vanilla = evaluate_by_subset_attacked(config_vanilla, return_cm=True)
    eval_results = {'Vanilla': (metrics_vanilla, cm_vanilla)}
    for model_name, cavs in all_cavs.items():
        metrics, cm = evaluate_by_subset_attacked(
            config_pclarc,
            cav=cavs[concept_id].unsqueeze(0),
            return_cm=True
        )
        eval_results[model_name] = (metrics, cm)
    
    data = []
    ## prepare data for plotting
    # for metric_name, value in accuracy_metrics_vanilla.items():
    #     data.append({'Metric': metric_name, 'Value': value, 'Model': 'Vanilla'})
    # for metric_name, value in accuracy_metrics_pclarc_baseline.items():
    #     data.append({'Metric': metric_name, 'Value': value, 'Model': 'Baseline CAV'})
    # for metric_name, value in accuracy_metrics_pclarc_orthogonal.items():
    #     data.append({'Metric': metric_name, 'Value': value, 'Model': 'Orthogonal CAV'})
    for model_name, (metrics, _) in eval_results.items():
        for metric_name, value in metrics.items():
            data.append({'Metric': metric_name, 'Value': value, 'Model': model_name})

    df = pd.DataFrame(data)

    # Filter the DataFrame to include only relevant metrics
    selected_metrics = [
        'test_accuracy_ch', 'test_accuracy_attacked', 'test_accuracy_clean',
        'test_fpr_1_ch', 'test_fpr_1_attacked', 'test_fpr_1_clean',
        'test_fnr_1_ch', 'test_fnr_1_attacked', 'test_fnr_1_clean'
    ]

    df_filtered = df[df['Metric'].isin(selected_metrics)]

    # Create a new column to categorize metrics for plotting
    df_filtered['Category'] = df_filtered['Metric'].str.extract(r'_(clean|attacked|ch)')[0]
    df_filtered['Metric Type'] = df_filtered['Metric'].str.replace(r'_(1|ch|attacked|clean)', '', regex=True)
    df_filtered = df_filtered.loc[~(df_filtered['Category'] == "ch")]

    savename = f"{savepath}/metric_comparison_{config['config_name']}_{cav_type}"
    plot_metric_comparison(df_filtered, savename)

    # for cm, name in [
    #     (cm_vanilla, "vanilla"),
    #     (cm_pclarc_baseline, "pclarc_baseline"),
    #     (cm_pclarc_orthogonal, "pclarc_orthogonal"),
    #     ]:
    #     savename = f"{savepath}/confusion_matrix_{name}_{config['config_name']}_{cav_type}"
    #     plot_confusion_matrices(cm["test"], savename)
    for model_name, (_, cm) in eval_results.items():
        savename = f"{savepath}/confusion_matrix_{model_name.replace(' ', '_').lower()}_{config['config_name']}_{cav_type}"
        plot_confusion_matrices(cm['test'], savename)

    print("Done, log to wandb")

    if config.get('wandb_api_key', None):
        results_to_log = {}
        # for name, metrics in [("Vanilla", accuracy_metrics_vanilla),
        #                       ("PClarc_baseline", accuracy_metrics_pclarc_baseline),
        #                       ("PClarc_orthogonal", accuracy_metrics_pclarc_orthogonal),
        #                       ]:
        for model_name, (metrics, _) in eval_results.items():
            for m in selected_metrics:
                # results_to_log[f"{m}_{name}_{cav_type}"] = metrics[m]
                results_to_log[f"{m}_{model_name}_{cav_type}"] = metrics[m]

        wandb.log(results_to_log)
        # [wandb.log({f"metric_comparison_{m}_{cav_type}": wandb.Image(f"{savepath}/metric_comparison_{config['config_name']}_{cav_type}_{m}.jpg")}) for m in ["test_accuracy", "test_fnr"]]
        # wandb.log({f"concept_similarity_after_pclarc_{cav_type}": wandb.Image(f"{savepath}/concept_similarity_after_pclarc_{config['config_name']}_{cav_type}.jpg")})
        # # [wandb.log({f"cm_{method}": wandb.Image(f"{savepath}/confusion_matrix_{method}_{config['config_name']}.jpg")}) for method in [
        # #     "vanilla", "pclarc_baseline", "pclarc_orthogonal"
        # # ]]
        # Log concept similarity and metric comparison plots
        wandb.log({
            f"concept_similarity_after_pclarc_{cav_type}": wandb.Image(
                f"{savepath}/concept_similarity_after_pclarc_{config['config_name']}_{cav_type}.jpg"
            )
        })
        [wandb.log({
            f"metric_comparison_{config['config_name']}_{cav_type}_{metric_type}": wandb.Image(
                f"{savepath}/metric_comparison_{config['config_name']}_{cav_type}_{metric_type}.jpg"
            )
        }) for metric_type in ["test_accuracy", "test_fnr"]]

        # Log confusion matrices for each CAV variant
        for model_name in eval_results:
            wandb.log({
                f"confusion_matrix_{model_name.replace(' ', '_').lower()}_{config['config_name']}_{cav_type}": wandb.Image(
                    f"{savepath}/confusion_matrix_{model_name.replace(' ', '_').lower()}_{config['config_name']}_{cav_type}.jpg"
                )
            })


    

def plot_metric_comparison(df, savename):
    plt.rcParams.update({'font.size': 9, 'legend.fontsize': 9, 'axes.titlesize': 11})

    metric_names = {
        "test_accuracy": "Accuracy",
        "test_fnr": "False Positive Rate",
    }
    for metric_type in ["test_accuracy", "test_fnr"]:
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(6, 4))
        sns.barplot(x='Category', y='Value', hue='Model', 
                    data=df[df["Metric Type"] == metric_type])
        plt.ylabel(metric_names[metric_type])
        plt.xlabel("")
        if metric_type == "test_accuracy":
            plt.ylim(0.5, 0.95)
            plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        # plt.title("")
        # plt.title(metric_names[metric_type])
        plt.legend(title='', loc='upper left', bbox_to_anchor=(-.22, -.15),ncols=3)
        plt.show()
        [fig.savefig(f"{savename}_{metric_type}.{ending}", bbox_inches="tight", dpi=500) for ending in ["png", "jpg", "pdf"]]
        plt.close()

def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return accuracy, false_positive_rate, false_negative_rate, recall, precision

def plot_confusion_matrices(confusion_matrices, savename):
    # Calculate accuracy for attacked and clean
    metrics_attacked = calculate_metrics(confusion_matrices['attacked'])
    metrics_clean = calculate_metrics(confusion_matrices['clean'])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot for 'attacked'
    sns.heatmap(confusion_matrices['attacked'], 
                annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
                xticklabels=['Others', 'Blonde'],
                yticklabels=['Others', 'Blonde'])
    axes[0].set_title(f'Confusion Matrix (Attacked)\n'
                    f'Accuracy: {metrics_attacked[0]:.2f}\n'
                    f'False Positive Rate: {metrics_attacked[1]:.2f}\n'
                    f'False Negative Rate: {metrics_attacked[2]:.2f}\n'
                    f'Recall: {metrics_attacked[3]:.2f}\n'
                    f'Precision: {metrics_attacked[4]:.2f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Plot for 'clean'
    sns.heatmap(confusion_matrices['clean'], 
                annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1],
                xticklabels=['Others', 'Blonde'],
                yticklabels=['Others', 'Blonde'])
    axes[1].set_title(f'Confusion Matrix (Clean)\n'
                    f'Accuracy: {metrics_clean[0]:.2f}\n'
                    f'False Positive Rate: {metrics_clean[1]:.2f}\n'
                    f'False Negative Rate: {metrics_clean[2]:.2f}\n'
                    f'Recall: {metrics_clean[3]:.2f}\n'
                    f'Precision: {metrics_clean[4]:.2f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()
    [fig.savefig(f"{savename}.{ending}", bbox_inches="tight") for ending in ["png", "jpg"]]
    plt.close()

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def get_activations_ds(model, dataset, config, device, split):
    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    dataset_split = dataset.get_subset_by_idxs(sets[split])
    dl_split = DataLoader(dataset_split, batch_size=16, shuffle=False)

    # Register forward hook for layer of interest
    handles = []
    layer = config["layer_name"]
    for n, m in model.named_modules():
        if n.endswith(layer):
            handles.append(m.register_forward_hook(get_activation))

    activations_split = None
    for x, y in tqdm.tqdm(dl_split):
        _ = model(x.to(device))
        acts_batch = activations.clone().detach().cpu().flatten(start_dim=2).max(dim=2).values
        activations_split = acts_batch if activations_split is None else torch.cat([activations_split, acts_batch])
          
          
    [h.remove() for h in handles]
    return activations_split, dataset_split.metadata

def plot_concept_similarities(df_similarities, savename):
    interesting_concepts = [
        "timestamp", 
        "box", 
        "Bangs", 
        "Blond_Hair", 
        "Wearing_Necklace", 
        "Pointy_Nose",
        "Rosy_Cheeks", 
        # "High_Cheekbones", 
        # "Smiling", 
        # "Black_Hair", 
        # "Young", 
        # "Wearing_Necklace", 
        # "High_Cheekbones"
    ]

    fig = plt.figure(figsize=(8, 3))
    plt.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle='--')  # Add this line for the horizontal line

    sns.boxplot(df_similarities.loc[df_similarities["concept"].isin(interesting_concepts)], x="concept", y="v", hue="model")
    plt.xticks(rotation=30)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.ylabel("Cos. sim. CAV/acts (w concept)")
    plt.show()
    [fig.savefig(f"{savename}.{ending}", bbox_inches="tight") for ending in ["png", "jpg"]]
    plt.close()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()