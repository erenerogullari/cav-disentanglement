import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import io
import os
import pandas as pd
from PIL import Image
from utils.sim_matrix import reorder_similarity_matrix, visualize_before_after_sim_matrices 


def visualize_sim_matrix_gif(A: torch.Tensor, concepts: List, auc_scores_hist: np.ndarray, labels: List[str], title: str = 'Similarity Matrix', t: int = 0, auc_min=0.5, auc_max=1):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns for side-by-side plots

    # Plot the similarity matrix
    sns.heatmap(A.cpu().numpy(), annot=False, cmap='coolwarm', cbar=True,
                vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels, ax=ax[0])
    ax[0].set_title(title)
    ax[0].tick_params(axis='x', rotation=90)

    # Compute AuC scores
    performance_all = auc_scores_hist.mean(1)
    auc_change = np.abs(auc_scores_hist[0,:] - auc_scores_hist[-1,:])
    most_unstable_id = np.argmax(auc_change)
    most_stable_id = np.argmin(auc_change)
    perf_most_unstable = auc_scores_hist[:, most_unstable_id]
    perf_most_stable = auc_scores_hist[:, most_stable_id]

    # Plot the AUC curve
    performance_subset = performance_all[:t+1]  # Use indices [0, t] for plotting
    perf_most_unstable_subset = perf_most_unstable[:t + 1]    # Use indices [0, t] for plotting
    perf_most_stable_subset = perf_most_stable[:t + 1]    # Use indices [0, t] for plotting
    ax[1].plot(range(len(performance_subset)), performance_subset, color='red', linewidth=3, label='Average AUC')
    ax[1].plot(range(len(perf_most_unstable_subset)), perf_most_unstable_subset, color='blue', linewidth=2, linestyle='--', label=f'Most Unstable Concept: {concepts[most_unstable_id]}')
    ax[1].plot(range(len(perf_most_stable_subset)), perf_most_stable_subset, color='green', linewidth=2, linestyle='--', label=f'Most Stable Concept: {concepts[most_stable_id]}')
    ax[1].set_title("AUC Evolution")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("AUC")
    ax[1].set_xlim([0, len(performance_all)])  # Set x-limits to max length
    ax[1].set_ylim([auc_min, auc_max])
    ax[1].grid(axis='y', linestyle='--', linewidth=0.5)
    ax[1].legend()

    plt.tight_layout()  # Adjust spacing between subplots

def create_similarity_matrix_gif(cos_sims, concepts, auc_scores_hist, filename='cos_sim_evolution.gif', order=None, duration=500, auc_min=0.5, auc_max=1):
    images = []
    filename = 'media/' + filename

    for i, cos_sim_matrix in enumerate(cos_sims):
        title = f'Similarity Matrix at Step {i * 10}'

        if i == 0 and order is None:
            A_ordered, order = reorder_similarity_matrix(cos_sim_matrix.cpu().numpy())
            labels_ordered = [concepts[idx] for idx in order]
            A_ordered_tensor = torch.tensor(A_ordered)
        else:
            A = cos_sim_matrix.cpu().numpy()
            A_ordered = A[order, :][:, order]
            labels_ordered = [concepts[idx] for idx in order]
            A_ordered_tensor = torch.tensor(A_ordered)

        # Visualize both the similarity matrix and AUC curve
        visualize_sim_matrix_gif(A_ordered_tensor, concepts, auc_scores_hist, labels_ordered, title=title, t=i, auc_min=auc_min, auc_max=auc_max)

        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        # Load the image from the buffer
        img = Image.open(buf)
        img.load()  # Ensure the image data is read into memory
        images.append(img)
        buf.close()

    # Save the images as a GIF
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved as {filename}")


def plot_training_loss(cav_loss_history, orthogonality_loss_history, save_path=None):
    """
    Plots the classification loss and orthogonality loss over epochs.

    Args:
        epochs_range (range): Range of epochs (e.g., range(1, N_EPOCHS + 1)).
        cav_loss_history (list): List of classification losses for each epoch.
        orthogonality_loss_history (list): List of orthogonality losses for each epoch.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    plt.figure(figsize=(10, 6))
    epochs_range = range(len(cav_loss_history))
    plt.plot(epochs_range, cav_loss_history, color='r', label='CAV Loss')
    plt.plot(epochs_range, orthogonality_loss_history, color='c', label='Orthogonality Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Time')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_metrics_over_time(epochs_logged, cav_performance_history, avg_precision_hist, cav_uniqueness_history, threshold=None, early_exit_epoch=None, save_path=None):
    """
    Plots AUC, Uniqueness, and Avg Precision over time.

    Args:
        epochs_logged (list): List of epochs where metrics are logged.
        cav_performance_history (list): List of mean AUC scores.
        avg_precision_hist (list): List of average precision scores.
        cav_uniqueness_history (list): List of uniqueness scores.
        threshold (float): AUC threshold line to display.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_logged, cav_performance_history, color='b', linestyle='-', label='Mean AUC')
    plt.plot(epochs_logged, avg_precision_hist, color='c', linestyle='-', label='Avg Precision')
    plt.plot(epochs_logged, cav_uniqueness_history, color='orange', linestyle='-.', label='Uniqueness')

    # Threshold annotation
    if threshold is not None:
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
        plt.text(epochs_logged[-1], threshold - 0.02, f"AuC Threshold: {threshold}", color="gray", fontsize=10, va="center", ha="right")

    # Early exit visualization: red vertical dashed line with red label below the x-axis
    if early_exit_epoch is not None:
        plt.axvline(x=early_exit_epoch, color='red', linestyle='--')
        ax = plt.gca()
        
        # Color the x-axis tick label corresponding to early_exit_epoch in red
        for tick in ax.get_xticklabels():
            try:
                if float(tick.get_text()) == float(early_exit_epoch):
                    tick.set_color("red")
            except ValueError:
                continue

    # Set y-limits
    y_min = min(min(cav_performance_history), min(avg_precision_hist), min(cav_uniqueness_history))
    plt.ylim(max(0, y_min - 0.1), 1.1)

    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.title('AUC, Uniqueness and Avg Precision over Time')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_cosine_similarity(cos_sim_matrix_original, cos_sim_matrix, concepts, save_path=None):
    """
    Visualizes the cosine similarity matrices before and after training.

    Args:
        cos_sim_matrix_original (torch.Tensor): Original cosine similarity matrix.
        cos_sim_matrix (torch.Tensor): Cosine similarity matrix after training.
        concepts (list): List of concept names for labeling.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    cos_sim_before_after = visualize_before_after_sim_matrices(cos_sim_matrix_original, cos_sim_matrix, concepts)
    if save_path:
        cos_sim_before_after.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        cos_sim_before_after.show()


def plot_auc_before_after(auc_before, auc_after, concepts, save_path=None):
    """
    Plots AUC scores before and after training for each concept.

    Args:
        auc_before (list): List of AUC scores before training.
        auc_after (list): List of AUC scores after training.
        concepts (list): List of concept names.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    auc_diff = np.array(auc_after) - np.array(auc_before)
    sorted_indices_auc = np.argsort(auc_diff)
    sorted_concepts_auc = [concepts[i] for i in sorted_indices_auc]
    sorted_auc_before = [auc_before[i] for i in sorted_indices_auc]
    sorted_auc_after = [auc_after[i] for i in sorted_indices_auc]

    X_axis = np.arange(len(auc_before))
    plt.figure(figsize=(12, 6))
    plt.plot(X_axis, sorted_auc_before, marker='o', label='AuC Before', linestyle='-', markersize=8)
    plt.plot(X_axis, sorted_auc_after, marker='o', label='AuC After', linestyle='-', markersize=8)
    plt.xticks(X_axis, sorted_concepts_auc, rotation=90)
    plt.xlabel('Concepts')
    plt.ylabel('AuC Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='x')
    plt.title('AuC of CAVs per Concept')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uniqueness_before_after(uniqueness_before, uniqueness_after, concepts, save_path=None):
    """
    Plots uniqueness scores before and after training for each concept.

    Args:
        uniqueness_before (list): List of uniqueness scores before training.
        uniqueness_after (list): List of uniqueness scores after training.
        concepts (list): List of concept names.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    unq_diff = np.array(uniqueness_after) - np.array(uniqueness_before)
    sorted_indices_unq = np.argsort(unq_diff)
    sorted_concepts_unq = [concepts[i] for i in sorted_indices_unq]
    sorted_uniqueness_before = [uniqueness_before[i] for i in sorted_indices_unq]
    sorted_uniqueness_after = [uniqueness_after[i] for i in sorted_indices_unq]

    X_axis = np.arange(len(uniqueness_before))
    plt.figure(figsize=(12, 6))
    plt.plot(X_axis, sorted_uniqueness_before, marker='o', label='Uniqueness Before', linestyle='-', markersize=8)
    plt.plot(X_axis, sorted_uniqueness_after, marker='o', label='Uniqueness After', linestyle='-', markersize=8)
    plt.xticks(X_axis, sorted_concepts_unq, rotation=90)
    plt.xlabel('Concepts')
    plt.ylabel('Uniqueness Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='x')
    plt.title('Uniqueness of CAVs per Concept')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_confusion_trajectories(confusion_matrices, title='Confusion Matrix Trajectories', save_path=None, normalize=True):
    """
    Visualizes average TP, TN, FP, FN confusion trajectories together with 95% confidence interval over epochs.
    Args:
        confusion_matrices (list): List of confusion matrices for each concept per epoch.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown.
        normalize (bool): Whether to normalize the confusion matrix values.
    """
    records = []
    for epoch_idx, cm in enumerate(confusion_matrices):
        n_concepts = cm.shape[0]
        totals = cm.sum(axis=(1,2)) if normalize else np.ones(n_concepts)
        for c in range(n_concepts):
            t = totals[c]
            records.extend([
                {'epoch': epoch_idx*10, 'concept': c, 'ratio': 'TP', 'value': cm[c,0,0]/t},
                {'epoch': epoch_idx*10, 'concept': c, 'ratio': 'TN', 'value': cm[c,1,1]/t},
                {'epoch': epoch_idx*10, 'concept': c, 'ratio': 'FP', 'value': cm[c,0,1]/t},
                {'epoch': epoch_idx*10, 'concept': c, 'ratio': 'FN', 'value': cm[c,1,0]/t},
            ])
    df = pd.DataFrame(records)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x='epoch', y='value', hue='ratio', markers=True, dashes=False)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Normalized Counts' if normalize else 'Counts')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()