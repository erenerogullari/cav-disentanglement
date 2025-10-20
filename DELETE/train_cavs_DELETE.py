import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import copy
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pickle
import os
from models import SignalCAV, LinearCAV
from utils.cav import compute_all_cavs
from utils.metrics import get_accuracy, get_avg_precision, get_uniqueness, compute_auc_performance, get_auconf, get_confusion_matrices
from utils.sim_matrix import reorder_similarity_matrix
from utils.visualizations import plot_training_loss, plot_metrics_over_time, plot_cosine_similarity, plot_auc_before_after, plot_uniqueness_before_after, visualize_confusion_trajectories

init_dict = {
    "models.SignalCAV": "checkpoints/scav_vgg16_celeba.pth",
    "models.LinearCAV": "checkpoints/lcav_vgg16_celeba.pth",
}

def name_model(config):
    if  config.cav.model._target_ == "models.SignalCAV":
        model_name = f"cavs-signal:alpha{config.cav.alpha}"
    elif config.cav.model._target_ == "models.LinearCAV":
        model_name = f"cavs-linear:alpha{config.cav.alpha}"
    else:
        raise ValueError(f"Unknown CAV model: {config.cav.model._target_}")

    if config.cav.beta is not None:
        model_name += f"_beta{config.cav.beta}_n_targets{config.cav.n_targets}"

    # model_name += f"_lr{config.train.learning_rate}_diffae"
    model_name += f"_lr{config.train.learning_rate}"

    return model_name

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

def train_test_split(tensor_dataset, train_ratio):
    total_size = len(tensor_dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])
    print(f"Total samples: {total_size}")
    print(f"Training samples: {train_size}")
    print(f"Testing samples: {test_size}")
    return train_dataset, test_dataset

def initialize_weights(C, labels, alpha, beta, n_targets, device):
    if alpha == 0:
        return 0
    
    weights = alpha * torch.ones_like(C, device=device)

    if beta != None:
        # Extract the rows and cols
        similarities = torch.triu(C.abs(), diagonal=1).clone()
        sorted_indices = torch.argsort(similarities.flatten(), descending=True)
        rows = sorted_indices // similarities.size(1)
        cols = sorted_indices % similarities.size(1)
        
        # Iterate through ordered pairs and select the first k valid ones
        selected_pairs = []
        for i, j in zip(rows.tolist(), cols.tolist()):
            if len(set(torch.where(labels[:, i] == 1)[0].tolist()).intersection(
                   set(torch.where(labels[:, j] == 1)[0].tolist()))) != 0:
                selected_pairs.append((i, j))
                
            if len(selected_pairs) == n_targets:
                break
        
        # Assign weights for the selected pairs
        for i, j in selected_pairs:
            weights[i, j] = np.sqrt(beta)
            weights[j, i] = np.sqrt(beta)

    return weights

def save_results(cavs, metrics, save_dir):
    cavs_normalized = cavs / torch.norm(cavs, dim=1, keepdim=True)
    torch.save(cavs_normalized, f'{save_dir}/cavs.pt')

    # Save metrics
    with open(f"{save_dir}/metrics/auc_hist.pkl", "wb") as f:
        pickle.dump(metrics['auc_hist'], f)

    with open(f"{save_dir}/metrics/uniqueness_hist.pkl", "wb") as f:
        pickle.dump(metrics['uniqueness_hist'], f)

    with open(f"{save_dir}/metrics/precision_hist.pkl", "wb") as f:
        pickle.dump(metrics['precision_hist'], f)

    with open(f"{save_dir}/metrics/confusion_matrix_hist.pkl", "wb") as f:
        pickle.dump(metrics['confusion_matrix_hist'], f)

    with open(f"{save_dir}/metrics/cav_loss_hist.pkl", "wb") as f:
        pickle.dump(metrics['cav_loss_hist'], f)

    with open(f"{save_dir}/metrics/orth_loss_hist.pkl", "wb") as f:
        pickle.dump(metrics['orth_loss_hist'], f)

    print('Model and results saved.')

def save_plots(cavs, cavs_original, metrics, x_latent, labels, concepts, save_dir):
    # Plot Training and Orthogonality Loss
    plot_training_loss( 
        cav_loss_history=metrics['cav_loss_hist'], 
        orthogonality_loss_history=metrics['orth_loss_hist'], 
        save_path=f"{save_dir}/media/training_loss.png"
    )
    
    # Plot Metrics History
    cav_performance_history = np.mean(np.array(metrics['auc_hist']), axis=1)
    cav_uniqueness_history = np.mean(np.array(metrics['uniqueness_hist']), axis=1)
    epochs_logged = [10*i for i in range(len(cav_performance_history))]
    plot_metrics_over_time(
        epochs_logged=epochs_logged,
        cav_performance_history=cav_performance_history,
        avg_precision_hist=metrics['precision_hist'],
        cav_uniqueness_history=cav_uniqueness_history,
        threshold=None,
        early_exit_epoch=metrics['early_exit_epoch'],
        save_path=f"{save_dir}/media/metrics_plot.png"
    )

    # Plot Cosine Sim Before/After
    cavs_normalized = cavs / torch.norm(cavs, dim=1, keepdim=True)
    cos_sim_matrix = cavs_normalized @ cavs_normalized.T
    cavs_original = cavs_original.detach().cpu()
    cos_sim_matrix_original = cavs_original @ cavs_original.T 
    plot_cosine_similarity(
        cos_sim_matrix_original=cos_sim_matrix_original,
        cos_sim_matrix=cos_sim_matrix,
        concepts=concepts,  # List of concept names
        save_path=f"{save_dir}/media/cos_sim_before_after.png"
    )

    # Plot AuC Before/After
    auc_before = compute_auc_performance(cavs_original, x_latent, labels)
    auc_after = compute_auc_performance(cavs_normalized, x_latent, labels)
    auc_diff = np.array(auc_after) - np.array(auc_before)
    sorted_indices_auc = np.argsort(auc_diff)
    sorted_concepts_auc = [concepts[i] for i in sorted_indices_auc]
    sorted_auc_before = [auc_before[i] for i in sorted_indices_auc]
    sorted_auc_after = [auc_after[i] for i in sorted_indices_auc]
    plot_auc_before_after(
        auc_before=sorted_auc_before,
        auc_after=sorted_auc_after,
        concepts=sorted_concepts_auc,  # List of concept names
        save_path=f"{save_dir}/media/auc_before_after.png"
    )

    # Uniqueness Before/After
    uniqueness_before = get_uniqueness(cos_sim_matrix_original)
    uniqueness_after = get_uniqueness(cos_sim_matrix)

    # Sort Uniqueness changes
    unq_diff = np.array(uniqueness_after) - np.array(uniqueness_before)
    sorted_indices_unq = np.argsort(unq_diff)
    sorted_concepts_unq = [concepts[i] for i in sorted_indices_unq]
    sorted_uniqueness_before = [uniqueness_before[i] for i in sorted_indices_unq]
    sorted_uniqueness_after = [uniqueness_after[i] for i in sorted_indices_unq]

    plot_uniqueness_before_after(
        uniqueness_before=sorted_uniqueness_before,
        uniqueness_after=sorted_uniqueness_after,
        concepts=sorted_concepts_unq, 
        save_path=f"{save_dir}/media/uniqueness_before_after.png"
    )

    # Plot Confusion Trajectories
    visualize_confusion_trajectories(metrics['confusion_matrix_hist'], save_path=f"{save_dir}/media/confusion_trajectories.png")

    print(f"Plots saved in '{save_dir}/media'.")


@hydra.main(version_base=None, config_path="../configs/train_cavs", config_name="config.yaml")
def main(config: DictConfig) -> None:

    # Set up the device and seed from the config
    device = config.train.device
    print(f"Using device: {device}")
    torch.manual_seed(config.train.random_seed)

    # Paths to data variables
    model_name = name_model(config)
    original_cwd = hydra.utils.get_original_cwd()
    latents_path = os.path.join(original_cwd, config.paths.latents)
    labels_path = os.path.join(original_cwd, config.paths.labels)
    concept_names_path = os.path.join(original_cwd, config.paths.concept_names)
    save_dir = os.path.join(original_cwd, "results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'media'), exist_ok=True)

    # Load the data variables
    x_latent = torch.load(latents_path, weights_only=True)      # Shape: (num_samples, n_features)
    labels = torch.load(labels_path, weights_only=True)         # Shape: (num_samples, n_concepts)
    labels = labels.to(torch.float32).clamp(min=0)
    with open(concept_names_path, "rb") as f:
        concepts_names = pickle.load(f)                                                                   

    # (Optional) Save the current config for reproducibility
    with open(os.path.join(save_dir, 'config.yaml'), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Prepare training data
    tensor_ds = TensorDataset(x_latent, labels)
    n_features, n_concepts = x_latent.shape[1], labels.shape[1]
    train_dataset, test_dataset = train_test_split(tensor_ds, config.train.train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)

    # Prepare test data
    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data).to(device)        
    test_labels = torch.stack(test_labels).to(device)

    # CAV Initialization
    cav_model = instantiate(config.cav.model, n_concepts=n_concepts, n_features=n_features)
    if config.cav.optimal_init:
        cav_model.load_state_dict(torch.load(init_dict[config.cav.model._target_], weights_only=True))
    cav_model = cav_model.to(device)

    # Debug
    # scavs_r2r, _ = cav_model.get_params()
    # print(scavs_r2r.norm(dim=1))
    # metrics = eval_epoch(test_data, test_labels, cav_model, device)
    # print(f"Initial AUC: {np.mean(metrics['auc_scores'])}")
    # print(f"Initial Uniqueness: {np.mean(metrics['uniqueness'])}")
    

    # Similarity matrix and weights initialization
    cavs_original = torch.load(init_dict[config.cav.model._target_], weights_only=True)["weights"].squeeze(0)
    cavs_original = cavs_original / torch.norm(cavs_original, dim=1, keepdim=True)  # Normalize the CAVs
    # cavs_original = torch.randn(n_concepts, n_features, device=device)  # Random initialization for testing
    C = cavs_original @ cavs_original.T
    _, order = reorder_similarity_matrix(C.detach().cpu().numpy())
    weights = initialize_weights(C, labels, config.cav.alpha, config.cav.beta, config.cav.n_targets, device=device)

    # Define optimizer
    optimizer = optim.Adam(cav_model.parameters(), lr=config.train.learning_rate)

    # Initialize metrics storage
    cav_loss_history = []
    orthogonality_loss_history = []
    auc_scores_history = []
    uniqueness_history = []
    avg_precision_hist = []
    confusion_matrix_history = []

    # Early exit variables
    best_uniqueness = 0.0
    uniqueness_epsilon = 0.01
    best_auc = 0.0
    auc_epsilon = 0.01
    best_cavs = None
    early_exit_epoch = 0

    ### MAIN LOOP ###
    for epoch in tqdm(range(config.train.num_epochs+1), desc="Epochs"):
        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            metrics = eval_epoch(test_data, test_labels, cav_model, device)
            auc_scores_history.append(metrics['auc_scores'])
            uniqueness_history.append(metrics['uniqueness'])
            avg_precision_hist.append(metrics['avg_precision'])
            confusion_matrix_history.append(metrics['confusion_matrix'])
            mean_auc = np.mean(metrics['auc_scores'])
            mean_uniqueness = np.mean(metrics['uniqueness'])

            if (config.cav.exit_criterion == "orthogonality" and mean_uniqueness > best_uniqueness + uniqueness_epsilon) or (config.cav.exit_criterion == "auc" and mean_auc > best_auc + auc_epsilon):
                best_uniqueness = mean_uniqueness
                best_auc = mean_auc
                early_exit_epoch = epoch
                best_cavs = cav_model.get_params()[0].detach()

                #! DELETE
                cav_model.save_state_dict(save_dir)

        # Train for one epoch
        epoch_cav_loss, epoch_orth_loss = train_epoch(train_loader, cav_model, weights, optimizer, device)
        cav_loss_history.append(epoch_cav_loss)
        orthogonality_loss_history.append(epoch_orth_loss)
        
        if epoch % 10 == 0:
            tqdm.write(f"CAV Loss:  {epoch_cav_loss:.4f} | Orth Loss: {epoch_orth_loss:.4f}")
            tqdm.write(f"AuC Score: {mean_auc:.4f} | Uniqueness: {mean_uniqueness:.4f}")


    # Save the results
    final_metrics = {
        'cav_loss_hist': cav_loss_history,
        'orth_loss_hist': orthogonality_loss_history,
        'auc_hist': auc_scores_history,
        'uniqueness_hist': uniqueness_history,
        'precision_hist': avg_precision_hist,
        'confusion_matrix_hist': confusion_matrix_history,
        'early_exit_epoch':  early_exit_epoch,
    }
    save_results(best_cavs, final_metrics, save_dir)
    save_plots(best_cavs, cavs_original, final_metrics, x_latent, labels, concepts_names, save_dir)


if __name__ == "__main__":
    main()