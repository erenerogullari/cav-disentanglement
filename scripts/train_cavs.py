import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import pickle
import copy
import yaml
import pickle
import os
import shutil
import argparse
from models.signal_cav import SignalCAV
from utils.cav import compute_all_cavs
from utils.metrics import get_accuracy, get_avg_precision, get_uniqueness, compute_auc_performance
from utils.sim_matrix import reorder_similarity_matrix
from utils.visualizations import plot_training_loss, plot_metrics_over_time, plot_cosine_similarity, plot_auc_before_after, plot_uniqueness_before_after


# Define command-line arguments
parser = argparse.ArgumentParser(description="Train CAVs with configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
parser.add_argument("--latents", type=str, required=True, help="Path to the latent variables (.pt file).")
parser.add_argument("--labels", type=str, required=True, help="Path to the labels (.pt file).")
parser.add_argument("--concepts", type=str, required=True, help="Path to the concept names (.pkl file).")
parser.add_argument("--save_dir", type=str, required=True, help="Path to the save directory.")
args = parser.parse_args()

# Load config file
def load_config(config_path):
    _, ext = os.path.splitext(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config(args.config)

# Dynamically pick the directory name for saving the results
model_name = f"cavs:alpha{config['cav']['alpha']}_beta{config['cav']['beta']}_lr{config['train']['learning_rate']}" if config['cav']['beta'] != 'None' else f"cavs:alpha{config['cav']['alpha']}_lr{config['train']['learning_rate']}"
save_dir = os.path.join(args.save_dir, model_name)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'media'), exist_ok=True)
shutil.copy(args.config, os.path.join(save_dir, 'config.yaml'))

def train_epoch(dataloader, cav_model, weights, optimizer, device):
    n_concepts, n_features = cavs.shape[-2], cavs.shape[-1]
    
    epoch_cav_loss = 0.0
    epoch_orthogonality_loss = 0.0
    for x_batch, labels_batch in dataloader:
        optimizer.zero_grad()

        # Move data to device
        x_batch = x_batch.to(device)
        labels_batch = labels_batch.to(device)

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

def eval_epoch(test_data, test_labels, cav_model):
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
        probs = logits.softmax(dim=1)

        # Compute metrics
        metrics['accuracy'] = get_accuracy(probs, test_labels)
        metrics['avg_precision'] = get_avg_precision(probs, test_labels)

        # Compute AUC for individual concepts
        metrics['auc_scores'] = compute_auc_performance(cavs, test_data, test_labels)

        # Uniqueness and Cosine Sim
        metrics['uniqueness'] = get_uniqueness(cavs)
        metrics['cos_sim_matrix'] = (cavs_normalized @ cavs_normalized.T).cpu()
    
    test_data = test_data.cpu()
    
    return metrics

def train_test_split(tensor_dataset):
    total_size = len(tensor_dataset)
    train_size = int(total_size * config['train']['train_ratio'])
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])
    print(f"Total samples: {total_size}")
    print(f"Training samples: {train_size}")
    print(f"Testing samples: {test_size}")
    return train_dataset, test_dataset

def initialize_weights(C, labels):
    weights = config['cav']['alpha'] * torch.ones_like(C, device=device)
    
    beta = config['cav']['beta']
    if beta != 'None':
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
                
            if len(selected_pairs) == config['cav']['n_targets']:
                break
        
        # Assign weights for the selected pairs
        for i, j in selected_pairs:
            weights[i, j] = np.sqrt(beta)
            weights[j, i] = np.sqrt(beta)

    return weights

def save_results(cavs, metrics):
    cavs_normalized = cavs / torch.norm(cavs, dim=1, keepdim=True)
    torch.save(cavs_normalized, f'{save_dir}/cavs.pt')

    # Save metrics
    with open(f"{save_dir}/metrics/auc_hist.pkl", "wb") as f:
        pickle.dump(metrics['auc_hist'], f)

    with open(f"{save_dir}/metrics/uniqueness_hist.pkl", "wb") as f:
        pickle.dump(metrics['uniqueness_hist'], f)

    with open(f"{save_dir}/metrics/precision_hist.pkl", "wb") as f:
        pickle.dump(metrics['precision_hist'], f)

    with open(f"{save_dir}/metrics/cav_loss_hist.pkl", "wb") as f:
        pickle.dump(metrics['cav_loss_hist'], f)

    with open(f"{save_dir}/metrics/orth_loss_hist.pkl", "wb") as f:
        pickle.dump(metrics['orth_loss_hist'], f)

    print('Model and results saved.')

def save_plots(cavs, cavs_original, metrics, x_latent, labels, concepts):
    # Plot Training and Orthogonality Loss
    plot_training_loss(
        epochs_range=range(config['train']['num_epochs']+1), 
        cav_loss_history=metrics['cav_loss_hist'], 
        orthogonality_loss_history=metrics['orth_loss_hist'], 
        save_path=f"{save_dir}/media/training_loss.png"
    )
    
    # Plot Metrics History
    cav_performance_history = np.mean(np.array(metrics['auc_hist']), axis=1)
    cav_uniqueness_history = np.mean(np.array(metrics['uniqueness_hist']), axis=1)
    epochs_logged = list(range(0, config['train']['num_epochs']+1, 10))
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
        concepts=concepts,  # List of concept names
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
        concepts=concepts, 
        save_path=f"{save_dir}/media/uniqueness_before_after.png"
    )

    print(f"Plots saved in '{save_dir}/media'.")


if __name__ == "__main__":

    device = config['train']['device']
    print(f"Using device: {device}")
    torch.manual_seed(config['train']['random_seed'])

    # Variables
    x_latent = torch.load(args.latents, weights_only=True)      # Shape: (num_samples, n_features)
    labels = torch.load(args.labels, weights_only=True)         # Shape: (num_samples, n_concepts)
    labels = labels.clamp(min=0)
    with open(args.concepts, "rb") as f:
        concepts = pickle.load(f)                                                                   

    # Prepare training data
    tensor_ds = TensorDataset(x_latent, labels)
    n_features, n_concepts = x_latent.shape[1], labels.shape[1]
    train_dataset, test_dataset = train_test_split(tensor_ds)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'])

    # Prepare test data
    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data).to(device)        
    test_labels = torch.stack(test_labels).to(device) 

    # CAVs Initialization
    train_x_latent, train_labels = zip(*train_dataset)
    train_labels = torch.stack(train_labels)
    train_x_latent = torch.stack(train_x_latent)
    cavs_original = compute_all_cavs(train_x_latent.float(), train_labels.float())
    if config['cav']['optimal_init']:
        cavs = copy.deepcopy(cavs_original).to(device) 
    else:
        cavs = torch.randn(n_concepts, n_features)
    b = torch.randn(n_concepts, n_features)

    # CAV model
    cav_model = SignalCAV(cavs, b, device=device)

    # Similarity matrix and weights initialization
    C = cavs_original @ cavs_original.T
    _, order = reorder_similarity_matrix(C.cpu().numpy())
    weights = initialize_weights(C, labels)

    # Define optimizer
    optimizer = optim.SGD(cav_model.parameters(), lr=config['train']['learning_rate'])

    # Initialize metrics storage
    cav_loss_history = []
    orthogonality_loss_history = []
    auc_scores_history = []
    uniqueness_history = []
    avg_precision_hist = []

    # Early exit variables
    best_uniqueness = 0.0
    uniqueness_epsilon = 0.01
    best_cavs = None
    early_exit_epoch = 0

    ### MAIN LOOP ###
    for epoch in tqdm(range(config['train']['num_epochs']+1), desc="Epochs"):
        # Train for one epoch
        epoch_cav_loss, epoch_orth_loss = train_epoch(train_loader, cav_model, weights, optimizer, device)
        cav_loss_history.append(epoch_cav_loss)
        orthogonality_loss_history.append(epoch_orth_loss)
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            metrics = eval_epoch(test_data, test_labels, cav_model)
            auc_scores_history.append(metrics['auc_scores'])
            uniqueness_history.append(metrics['uniqueness'])
            avg_precision_hist.append(metrics['avg_precision'])
            mean_auc = np.mean(metrics['auc_scores'])
            mean_uniqueness = np.mean(metrics['uniqueness'])

            tqdm.write(f"CAV Loss:  {epoch_cav_loss:.4f} | Orth Loss: {epoch_orth_loss:.4f}")
            tqdm.write(f"AuC Score: {mean_auc:.4f} | Uniqueness: {mean_uniqueness:.4f}")

            if mean_uniqueness > best_uniqueness + uniqueness_epsilon:
                best_uniqueness = mean_uniqueness
                early_exit_epoch = epoch
                best_cavs, _ = cav_model.get_params()


    # Save the results
    final_metrics = {
        'cav_loss_hist': cav_loss_history,
        'orth_loss_hist': orthogonality_loss_history,
        'auc_hist': auc_scores_history,
        'uniqueness_hist': uniqueness_history,
        'precision_hist': avg_precision_hist,
        'early_exit_epoch':  early_exit_epoch,
    }
    save_results(best_cavs, final_metrics)
    save_plots(best_cavs, cavs_original, final_metrics, x_latent, labels, concepts)
