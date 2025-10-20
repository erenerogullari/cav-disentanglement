import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import os
import copy
from models import get_fn_model_loader
from datasets import get_dataset
from utils.cav import compute_cavs
from utils.metrics import get_accuracy, get_avg_precision, get_uniqueness, compute_auc_performance, get_auconf, get_confusion_matrices
from utils.sim_matrix import reorder_similarity_matrix
from experiments.lib.utils import name_experiment, initialize_weights, save_results, save_plots
from experiments.lib.activations import extract_latents

log = logging.getLogger(__name__)

def train_test_split(x_latent, labels, train_ratio):
    total_size = x_latent.shape[0]
    train_size = int(total_size * train_ratio)
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_data = x_latent[train_indices]
    train_labels = labels[train_indices]
    test_data = x_latent[test_indices]
    test_labels = labels[test_indices]
    return train_data, train_labels, test_data, test_labels

def train_epoch(dataloader, cav_model, weights, optimizer, device):
    
    epoch_cav_loss = 0.0
    epoch_orthogonality_loss = 0.0
    for x_batch, labels_batch in dataloader:
        optimizer.zero_grad()

        # Move data to device
        x_batch = x_batch.to(device)
        labels_batch = labels_batch.to(device).clamp(min=0)

        # Forward
        cav_loss, orthogonality_loss = cav_model.train_step(x_batch, labels_batch, weights)

        # Total Loss
        total_batch_loss = cav_loss + orthogonality_loss

        # Backpropagation
        total_batch_loss.backward()
        optimizer.step()

        epoch_cav_loss += cav_loss.item()
        epoch_orthogonality_loss += orthogonality_loss.item()

    avg_cav_loss = epoch_cav_loss / len(dataloader)
    avg_orthogonality_loss = epoch_orthogonality_loss / len(dataloader)
    
    return avg_cav_loss, avg_orthogonality_loss

def eval_epoch(test_data, test_labels, cav_model, device):
    cavs, _ = cav_model.get_params()
    n_concepts, n_features = cavs.shape[-2], cavs.shape[-1]
    test_data = test_data.to(device)
    cavs = cavs.to(device)
    metrics = {}
    
    with torch.no_grad():
        # Normalize CAVs
        cavs_normalized = cavs / torch.norm(cavs, dim=1, keepdim=True)

        # Predictions and labels for classification metrics
        x_normalized = test_data / test_data.norm(dim=1, keepdim=True) 
        logits = x_normalized @ cavs_normalized.T
        probs = logits.sigmoid()

        # Compute metrics
        test_labels = test_labels.to(device)
        metrics['accuracy'] = get_accuracy(probs, test_labels)
        metrics['avg_precision'] = get_avg_precision(probs, test_labels)

        # Confusion matrix
        metrics['confusion_matrix'] = get_auconf(probs, test_labels)

        # Compute AUC for individual concepts
        metrics['auc_scores'] = compute_auc_performance(cavs, test_data, test_labels)

        # Uniqueness and Cosine Sim
        metrics['uniqueness'] = get_uniqueness(cavs)
        metrics['cos_sim_matrix'] = (cavs_normalized @ cavs_normalized.T).cpu()
    
    test_data = test_data.cpu()
    
    return metrics

def train_cavs(cfg: DictConfig) -> nn.Module:
    """Train CAVs with disentanglement losses.
    Args:
        cfg (DictConfig): Configuration object.
    Returns:
        cav_model (nn.Module): Trained CAV model.
    """
    # Set up the device and seed from the config
    device = cfg.train.device
    log.info(f"Using device: {device}")
    torch.manual_seed(cfg.train.random_seed)
    model_name = name_experiment(cfg)
    original_cwd = hydra.utils.get_original_cwd()
    save_dir = os.path.join(original_cwd, "results", "disentangle_cavs", model_name)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'media'), exist_ok=True)

    # Load model, dataset and compute latents
    log.info(f"Loading model: {cfg.model.name}")
    model = get_fn_model_loader(cfg.model.name)(ckpt_path=cfg.model.ckpt_paths[cfg.dataset.name],
                                                pretrained=cfg.model.pretrained,
                                                n_class=cfg.model.n_class).to(device)

    log.info(f"Loading dataset: {cfg.dataset.name}")
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset_fn = get_dataset(dataset_name)
    dataset = dataset_fn(**dataset_cfg)
    dataset = get_dataset(cfg.dataset.name)(data_paths=cfg.dataset.data_paths,
                                        normalize_data=cfg.dataset.normalize_data,
                                        image_size=cfg.dataset.img_size)
    labels = dataset.get_labels().to(torch.float32).clamp(min=0)
    concept_names = dataset.get_concept_names()

    log.info(f"Extracting latent variables at layer: {cfg.cav.layer}")
    x_latent = extract_latents(cfg, model, dataset)
    n_concepts, n_features = labels.shape[1], x_latent.shape[1]
    train_latents, train_labels, test_latents, test_labels = train_test_split(x_latent, labels, cfg.train.train_ratio)
    train_dataset = TensorDataset(train_latents, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)

    # Initialize CAV model and weights (alpha)
    log.info(f"Initializing CAV model: {cfg.cav.name}")
    raw_cav_cfg = OmegaConf.to_container(cfg.cav, resolve=True)
    cav_cfg = {"_target_": raw_cav_cfg["_target_"]}
    cavs_original, bias_original = compute_cavs(train_latents, train_labels, type=cfg.cav.name, normalize=True)
    cav_model = instantiate(cav_cfg, n_concepts=n_concepts, n_features=n_features, device=device)
    if cfg.cav.optimal_init:
        cav_model.load_state_dict({'weights': cavs_original, 'bias': bias_original})
    cav_model = cav_model.to(device)
    C = cavs_original @ cavs_original.T
    _, order = reorder_similarity_matrix(C.detach().cpu().numpy())
    weights = initialize_weights(C, labels, cfg.cav.alpha, cfg.cav.beta, cfg.cav.n_targets, device=device)

    # Training metrics
    log.info("Starting training...")
    optimizer = optim.Adam(cav_model.parameters(), lr=cfg.train.learning_rate)
    cav_loss_history = []
    orthogonality_loss_history = []
    auc_scores_history = []
    uniqueness_history = []
    avg_precision_hist = []
    confusion_matrix_history = []
    best_uniqueness = 0.0
    uniqueness_epsilon = 0.01
    best_auc = 0.0
    auc_epsilon = 0.01
    best_cavs = copy.deepcopy(cav_model).to("cpu")
    early_exit_epoch = 0

    ### MAIN LOOP ###
    for epoch in tqdm(range(cfg.train.num_epochs+1), desc="Epochs"):
        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            metrics = eval_epoch(test_latents, test_labels, cav_model, device)
            auc_scores_history.append(metrics['auc_scores'])
            uniqueness_history.append(metrics['uniqueness'])
            avg_precision_hist.append(metrics['avg_precision'])
            confusion_matrix_history.append(metrics['confusion_matrix'])
            mean_auc = np.mean(metrics['auc_scores'])
            mean_uniqueness = np.mean(metrics['uniqueness'])

            if (cfg.cav.exit_criterion == "orthogonality" and mean_uniqueness > best_uniqueness + uniqueness_epsilon) or (cfg.cav.exit_criterion == "auc" and mean_auc > best_auc + auc_epsilon):
                best_uniqueness = mean_uniqueness
                best_auc = mean_auc
                early_exit_epoch = epoch
                best_cavs = copy.deepcopy(cav_model).to("cpu")

        # Train for one epoch
        epoch_cav_loss, epoch_orth_loss = train_epoch(train_loader, cav_model, weights, optimizer, device)
        cav_loss_history.append(epoch_cav_loss)
        orthogonality_loss_history.append(epoch_orth_loss)
        
        if epoch % 10 == 0:
            tqdm.write(f"CAV Loss:  {epoch_cav_loss:.4f} | Orth Loss: {epoch_orth_loss:.4f}")
            tqdm.write(f"AuC Score: {mean_auc:.4f} | Uniqueness: {mean_uniqueness:.4f}")


    # Save the results
    log.info(f"Training completed. Saving results to {save_dir}.")
    final_metrics = {
        'cav_loss_hist': cav_loss_history,
        'orth_loss_hist': orthogonality_loss_history,
        'auc_hist': auc_scores_history,
        'uniqueness_hist': uniqueness_history,
        'precision_hist': avg_precision_hist,
        'confusion_matrix_hist': confusion_matrix_history,
        'early_exit_epoch':  early_exit_epoch,
    }
    save_results(best_cavs.get_params()[0].detach(), final_metrics, save_dir)
    save_plots(best_cavs.get_params()[0].detach(), cavs_original, final_metrics, x_latent, labels, concept_names, save_dir)
    torch.save(best_cavs.state_dict(), os.path.join(save_dir, 'state_dict.pth'))

    return best_cavs