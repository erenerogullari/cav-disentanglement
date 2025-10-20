import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Sequence, Optional
import io
import os
import pandas as pd
from PIL import Image
from utils.sim_matrix import reorder_similarity_matrix 
from utils.heatmaps import _coerce_to_tensor, _normalize_batch, _heatmap_to_display_array
from crp.image import imgify


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


def visualize_heatmaps(
    image_tensor: torch.Tensor | np.ndarray,
    heatmaps,
    channel_avg: bool = False,
    conormalize: bool = True,
    suptitle: Optional[str] = None,
    titles: Optional[Sequence[str]] = None,
    dot_products: Optional[Sequence[float]] = None,
    fontsize: int = 14,
    display: bool = True,
):
    """Visualize heatmaps for a single concept alongside the source image.

    Args:
        image_tensor: Image tensor or array shaped (C, H, W) or (H, W, C).
        heatmaps: Tensor or sequence containing ``K`` heatmaps of shape
            (K, H, W) or (K, C, H, W). For a single heatmap you may pass an
            array shaped (H, W) or (C, H, W).
        channel_avg: Average the heatmaps across the channel dimension before
            visualization. Necessary when heatmaps have more than one channel.
        conormalize: If ``True`` normalize all heatmaps jointly by the maximum
            absolute activation; otherwise each heatmap is normalized
            independently.
        suptitle: Optional title placed on top of the figure.
        titles: Column titles. Must contain ``K + 1`` entries where the first
            title corresponds to the original image column.
        dot_products: Optional sequence of length ``K`` with scalar values to be
            annotated below each heatmap.
        fontsize: Font size for titles and annotations.
        display: When ``True`` the figure is shown. Otherwise it is closed but
            still returned to allow saving.

    Returns:
        matplotlib.figure.Figure: The constructed figure.
    """

    image = imgify(image_tensor)

    hm_tensor = _coerce_to_tensor(heatmaps).to(torch.float32)
    if hm_tensor.dim() == 2:
        hm_tensor = hm_tensor.unsqueeze(0)
    elif hm_tensor.dim() not in (3, 4):
        raise ValueError("`heatmaps` must have 2, 3 or 4 dimensions.")

    num_heatmaps = hm_tensor.shape[0]
    normalized = _normalize_batch(hm_tensor, channel_avg=channel_avg, conormalize=conormalize)

    column_titles = list(titles) if titles is not None else None
    if column_titles is None:
        column_titles = ['Original Image'] + [f'Heatmap {i + 1}' for i in range(num_heatmaps)]
    if len(column_titles) != num_heatmaps + 1:
        raise ValueError("`titles` must provide exactly one label per column.")

    dot_values = None
    if dot_products is not None:
        dot_tensor = _coerce_to_tensor(dot_products).to(torch.float32).flatten()
        if dot_tensor.numel() != num_heatmaps:
            raise ValueError("`dot_products` must match the number of heatmaps.")
        dot_values = dot_tensor.cpu().numpy()

    display_arrays = [
        _heatmap_to_display_array(normalized[idx], channel_avg=channel_avg)
        for idx in range(num_heatmaps)
    ]

    total_cols = num_heatmaps + 1
    fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes = axes.reshape(-1)
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(column_titles[0], fontsize=fontsize)

    for idx, heatmap_arr in enumerate(display_arrays, start=1):
        ax = axes[idx]
        ax.imshow(heatmap_arr, cmap='bwr', vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(column_titles[idx], fontsize=fontsize)
        if dot_values is not None:
            ax.text(
                0.95,
                0.05,
                f"Dot Product = {dot_values[idx - 1]:.2f}",
                transform=ax.transAxes,
                fontsize=fontsize - 2,
                ha='right',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    if display:
        plt.show()
    else:
        plt.close(fig)

    return fig

def visualize_heatmap_pair(
    image_tensor: torch.Tensor | np.ndarray,
    heatmaps,
    channel_avg: bool = False,
    conormalize: bool = True,
    suptitle: Optional[str] = None,
    titles: Optional[Sequence[str]] = None,
    dot_products=None,
    row_titles: Optional[Sequence[str]] = None,
    fontsize: int = 14,
    display: bool = True,
):
    """Visualize before/after heatmaps for multiple concepts.

    Args:
        image_tensor: Image tensor or array shaped (C, H, W) or (H, W, C).
        heatmaps: Tensor or nested sequence with shape (R, C, H, W) or
            (R, C, K, H, W). ``R`` represents the number of states (e.g.
            before/after) and ``C`` the number of concepts.
        channel_avg: Average each heatmap over the channel dimension before
            plotting.
        conormalize: If ``True`` normalize all heatmaps jointly using the
            global maximum absolute activation.
        suptitle: Optional string drawn above the figure.
        titles: Column titles (original image + one entry per concept).
        dot_products: Optional scalars with shape (R, C) (or broadcastable to
            that shape) for annotating each heatmap cell.
        row_titles: Labels for each row (defaults to ``['Before', 'After']`` if
            there are two rows).
        fontsize: Font size for titles and annotations.
        display: When ``True`` the figure is displayed, otherwise it is closed
            but still returned.

    Returns:
        matplotlib.figure.Figure: The constructed figure.
    """

    image = imgify(image_tensor)

    hm_tensor = _coerce_to_tensor(heatmaps).to(torch.float32)
    if hm_tensor.dim() not in (4, 5):
        raise ValueError("`heatmaps` must have 4 or 5 dimensions (rows, concepts, ...).")

    if hm_tensor.dim() == 4 and hm_tensor.shape[1] == 2 and hm_tensor.shape[0] != 2:
        hm_tensor = hm_tensor.permute(1, 0, 2, 3)
    if hm_tensor.dim() == 5 and hm_tensor.shape[1] == 2 and hm_tensor.shape[0] != 2:
        hm_tensor = hm_tensor.permute(1, 0, 2, 3, 4)

    n_rows = hm_tensor.shape[0]
    n_concepts = hm_tensor.shape[1]

    flat = hm_tensor.reshape(n_rows * n_concepts, *hm_tensor.shape[2:])
    normalized_flat = _normalize_batch(flat, channel_avg=channel_avg, conormalize=conormalize)
    normalized = normalized_flat.reshape(n_rows, n_concepts, *normalized_flat.shape[1:])

    column_titles = list(titles) if titles is not None else None
    if column_titles is None:
        column_titles = ['Original Image'] + [f'Concept {i + 1}' for i in range(n_concepts)]
    if len(column_titles) != n_concepts + 1:
        raise ValueError("`titles` must provide one entry per concept plus the original image column.")

    if row_titles is None:
        row_titles = ['Before', 'After'] if n_rows == 2 else [f'Row {i + 1}' for i in range(n_rows)]
    if len(row_titles) != n_rows:
        raise ValueError("`row_titles` length must match the number of rows (states).")

    dot_array = None
    if dot_products is not None:
        dp_tensor = _coerce_to_tensor(dot_products).to(torch.float32)
        if dp_tensor.dim() == 0:
            dp_tensor = dp_tensor.repeat(n_rows, n_concepts)
        elif dp_tensor.dim() == 1:
            if dp_tensor.numel() not in {n_concepts, n_rows}:
                raise ValueError("`dot_products` could not be broadcast to (rows, concepts).")
            if dp_tensor.numel() == n_concepts:
                dp_tensor = dp_tensor.unsqueeze(0).expand(n_rows, -1)
            else:
                dp_tensor = dp_tensor.unsqueeze(1).expand(-1, n_concepts)
        elif dp_tensor.dim() == 2:
            if dp_tensor.shape != (n_rows, n_concepts):
                if dp_tensor.shape == (n_concepts, n_rows):
                    dp_tensor = dp_tensor.transpose(0, 1)
                else:
                    raise ValueError("`dot_products` must match (rows, concepts) after broadcasting.")
        else:
            dp_tensor = dp_tensor.reshape(n_rows, n_concepts)
        dot_array = dp_tensor.cpu().numpy()

    display_arrays = [
        [
            _heatmap_to_display_array(normalized[row, col], channel_avg=channel_avg)
            for col in range(n_concepts)
        ]
        for row in range(n_rows)
    ]

    total_cols = n_concepts + 1
    fig, axes = plt.subplots(n_rows, total_cols, figsize=(4 * total_cols, 4 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for row in range(n_rows):
        for col in range(total_cols):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(image)
                ax.axis('off')
                if row == 0:
                    ax.set_title(column_titles[0], fontsize=fontsize)
            else:
                heatmap_arr = display_arrays[row][col - 1]
                ax.imshow(heatmap_arr, cmap='bwr', vmin=-1, vmax=1)
                ax.axis('off')
                if row == 0:
                    ax.set_title(column_titles[col], fontsize=fontsize)
                if dot_array is not None:
                    ax.text(
                        0.95,
                        0.05,
                        f"Dot Product = {dot_array[row, col - 1]:.2f}",
                        transform=ax.transAxes,
                        fontsize=fontsize - 2,
                        ha='right',
                        va='bottom',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                    )

        axes[row, -1].text(
            1.05,
            0.5,
            row_titles[row],
            transform=axes[row, -1].transAxes,
            fontsize=fontsize,
            ha='left',
            va='center',
            rotation=90
        )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    if display:
        plt.show()
    else:
        plt.close(fig)

    return fig


def visualize_sim_matrix(A: torch.Tensor, labels: List[str], title: str = 'Similarity Matrix', order:List = None, display:bool = False):
    """
   Visualizes a similarity matrix after grouping similar entries together.

    Parameters
    ----------
    A : ndarray of shape (N, N)
        A square similarity matrix where A[i, j] represents the similarity
        between the i-th and j-th vectors.
    labels: List of strings
        A list of label names that will be used for the plot.
    title : str
        Title of the plot.
    order : List[int], optional
        Specific order to arrange the matrix, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """

    if order is None:
        # Reorder the matrix
        A_ordered, order = reorder_similarity_matrix(A.cpu().numpy())
    else:
        A_ordered = A[order, :][:, order]

    labels_ordered = [labels[i] for i in order]

    # Visualize
    plt.figure(figsize=(12,10))
    sns.heatmap(A_ordered, annot=False, cmap='coolwarm', cbar=True, xticklabels=labels_ordered, yticklabels=labels_ordered, vmin=-1, vmax=1)
    plt.title(title)
    plt.xticks(rotation=90)
    
    # Return the plot
    fig = plt.gcf()
    if display:
        plt.show()
    if not display:
        plt.close(fig)  # Close the figure to prevent display

    return fig

def visualize_before_after_sim_matrices(
    before: torch.Tensor, 
    after: torch.Tensor, 
    labels: List[str],
    title: str = "Cosine Sim Matrices Before vs. After"
):
    """
    Visualize two cosine similarity matrices side by side with a shared colorbar.
    The labels are replaced with numbers, and a legend with numbers and labels is added below.

    Parameters:
    ----------
    before : torch.Tensor
        Original similarity matrix.
    after : torch.Tensor
        Optimized similarity matrix.
    labels : List[str]
        Labels for the axes.
    title : str
        Overall title for the figure.

    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing both matrices and the legend.
    """

    # Reorder both matrices using the same order as "before"
    before_np, order = reorder_similarity_matrix(before.cpu().numpy())
    after_np = after.cpu().numpy()[np.ix_(order, order)]

    # Create numbered labels
    n_labels = len(labels)
    numbered_labels = [str(i + 1) for i in range(n_labels)]
    ordered_labels = [labels[i] for i in order]

    # Plot setup
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), 
                             gridspec_kw={'height_ratios': [6, 0.5, 0.5], 'width_ratios': [1, 1]})
    cbar_ax = fig.add_axes([.93, .4, .02, .4])  # Shared colorbar position

    # Before matrix
    sns.heatmap(before_np, ax=axes[0, 0], cbar=True, cbar_ax=cbar_ax,
                cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=numbered_labels, yticklabels=numbered_labels)
    axes[0, 0].set_title("Before")
    axes[0, 0].set_xlabel("Concepts")
    axes[0, 0].set_ylabel("Concepts")

    # After matrix
    sns.heatmap(after_np, ax=axes[0, 1], cbar=False,
                cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=numbered_labels, yticklabels=numbered_labels)
    axes[0, 1].set_title("After")
    axes[0, 1].set_xlabel("Concepts")
    axes[0, 1].set_ylabel("Concepts")

    # Hide unused axes for legend placement
    for ax in axes[1:, :].ravel():
        ax.axis('off')

    # Prepare the legend data in two columns
    mid = (len(ordered_labels) + 1) // 2
    col1 = [f"{i + 1}. {ordered_labels[i]}" for i in range(mid)]
    col2 = [f"{i + 1}. {ordered_labels[i]}" for i in range(mid, len(ordered_labels))]
    if len(col1) > len(col2):
        col2.append('')

    # Add the two-column legend below the heatmaps
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    for i, (c1, c2) in enumerate(zip(col1, col2)):
        axes[2, 0].text(0, i * -0.5, c1, fontsize=12, ha='left')
        axes[2, 1].text(0, i * -0.5, c2, fontsize=12, ha='left')

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.3, hspace=0.0)

    plt.close(fig)  # Prevent display
    return fig


if __name__ == "__main__":
    # Quick smoke test that reuses existing CelebA CAV outputs to generate
    # preview figures. This mimics the localization script while exercising the
    # visualization helpers implemented above.
    try:
        from datasets import get_dataset
        from experiments.lib.localization import get_localization
        from models import get_fn_model_loader, get_canonizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Required visualization dependencies are missing. Make sure CRP/Zennit "
            "and the project modules are installed before running this test."
        ) from exc

    device = torch.device('cpu')
    experiment_name = "celeba-vgg16-linear_cav:alpha1_lr0.001"
    dataset_root = os.environ.get("CELEBA_ROOT", "/Users/erogullari/datasets/")
    results_dir = os.path.join("results", "disentangle_cavs", experiment_name)
    cav_path = os.path.join(results_dir, "cavs.pt")
    base_cav_path = os.path.join("checkpoints", "lcav_vgg16_celeba.pth")
    latent_path = os.path.join("variables", "latents_celeba_features.28_vgg16.pt")
    checkpoint_path = os.path.join("checkpoints", "checkpoint_vgg16_celeba.pth")

    if not os.path.exists(cav_path):
        raise FileNotFoundError(f"Disentangled CAVs not found at {cav_path}.")
    if not os.path.exists(base_cav_path):
        raise FileNotFoundError(f"Baseline CAVs not found at {base_cav_path}.")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent features not found at {latent_path}.")
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"CelebA root '{dataset_root}' is missing. Set CELEBA_ROOT to the dataset directory."
        )

    cavs_disentangled = torch.load(cav_path, map_location=device, weights_only=True)
    cavs_original = torch.load(base_cav_path, map_location=device, weights_only=True)["weights"]
    latents = torch.load(latent_path, map_location=device, weights_only=True)

    dataset = get_dataset("celeba")(data_paths=[dataset_root], normalize_data=True, image_size=224)
    if len(dataset) != latents.shape[0]:
        raise RuntimeError("Dataset size and latent cache size do not match."
                           " Regenerate latents before running the visualization test.")

    concept_names = dataset.get_concept_names()
    single_concept_id = 9 if len(concept_names) > 9 else 0
    single_concept_name = concept_names[single_concept_id]

    single_sample_id = int(dataset.sample_ids_by_concept[single_concept_name][0])
    single_image, _ = dataset[single_sample_id]
    single_batch = single_image.unsqueeze(0).to(device)

    model_loader = get_fn_model_loader("vgg16")
    model = model_loader(ckpt_path=checkpoint_path, pretrained=True, n_class=2).to(device)
    model.eval()
    canonizers = get_canonizer("vgg16")
    layer_name = "features.28"
    cav_mode = "max"

    heatmap_before = get_localization(
        cavs_original[single_concept_id].unsqueeze(0),
        single_batch,
        model,
        canonizers,
        layer_name,
        cav_mode=cav_mode,
        device=device,
    )[0]
    heatmap_after = get_localization(
        cavs_disentangled[single_concept_id].unsqueeze(0),
        single_batch,
        model,
        canonizers,
        layer_name,
        cav_mode=cav_mode,
        device=device,
    )[0]

    dot_before = torch.matmul(latents[single_sample_id], cavs_original[single_concept_id]).item()
    dot_after = torch.matmul(latents[single_sample_id], cavs_disentangled[single_concept_id]).item()

    single_titles = ["Original Image", "Original CAV", "Disentangled CAV"]
    single_fig = visualize_heatmaps(
        single_image,
        torch.stack([heatmap_before, heatmap_after]),
        suptitle=f"{single_concept_name} Localization",
        titles=single_titles,
        dot_products=[dot_before, dot_after],
        display=True,
    )

    concept_pair = (9, 38) if len(concept_names) > 38 else (0, 1)
    pair_names = [concept_names[idx] for idx in concept_pair]
    pair_candidates = set(dataset.sample_ids_by_concept[pair_names[0]]) & set(dataset.sample_ids_by_concept[pair_names[1]])
    if not pair_candidates:
        raise RuntimeError(f"No shared samples found for concept pair {pair_names}.")
    pair_sample_id = sorted(pair_candidates)[0]
    pair_image, _ = dataset[pair_sample_id]
    pair_batch = pair_image.unsqueeze(0).to(device)

    heatmaps_before_pair = []
    heatmaps_after_pair = []
    dots_before_pair = []
    dots_after_pair = []
    for cid in concept_pair:
        hm_before = get_localization(
            cavs_original[cid].unsqueeze(0),
            pair_batch,
            model,
            canonizers,
            layer_name,
            cav_mode=cav_mode,
            device=device,
        )[0]
        hm_after = get_localization(
            cavs_disentangled[cid].unsqueeze(0),
            pair_batch,
            model,
            canonizers,
            layer_name,
            cav_mode=cav_mode,
            device=device,
        )[0]
        heatmaps_before_pair.append(hm_before)
        heatmaps_after_pair.append(hm_after)
        dots_before_pair.append(torch.matmul(latents[pair_sample_id], cavs_original[cid]))
        dots_after_pair.append(torch.matmul(latents[pair_sample_id], cavs_disentangled[cid]))

    pair_heatmaps = torch.stack([
        torch.stack(heatmaps_before_pair),
        torch.stack(heatmaps_after_pair)
    ])
    pair_dot_products = torch.stack([
        torch.stack(dots_before_pair),
        torch.stack(dots_after_pair)
    ])

    pair_titles = ["Original Image"] + pair_names
    pair_fig = visualize_heatmap_pair(
        pair_image,
        pair_heatmaps,
        titles=pair_titles,
        row_titles=["Original CAV", "Disentangled CAV"],
        dot_products=pair_dot_products,
        display=True,
    )

    # preview_dir = os.path.join(results_dir, "media", "visualization_preview")
    # os.makedirs(preview_dir, exist_ok=True)
    # single_path = os.path.join(preview_dir, f"single_{single_concept_name.replace(' ', '_')}.png")
    # pair_path = os.path.join(preview_dir, f"pair_{pair_names[0].replace(' ', '_')}-{pair_names[1].replace(' ', '_')}.png")
    # single_fig.savefig(single_path, bbox_inches='tight')
    # pair_fig.savefig(pair_path, bbox_inches='tight')
    # print(f"Saved single-concept preview to {single_path}")
    # print(f"Saved concept-pair preview to {pair_path}")
