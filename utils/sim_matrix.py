import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
from typing import List

def reorder_similarity_matrix(S, method='average'):
    """
    Reorders a similarity matrix such that similar scores are clustered together.

    Parameters
    ----------
    S : ndarray of shape (N, N)
        A square similarity matrix where S[i, j] represents the similarity
        between the i-th and j-th vectors.

    method : str, optional, default='average'
        The linkage method to use for hierarchical clustering. Options include 'single',
        'complete', 'average', 'ward', etc.

    Returns
    -------
    S_reordered : ndarray of shape (N, N)
        The reordered similarity matrix with similar scores clustered together.

    order : ndarray of shape (N,)
        The order of the indices that was used to reorder the matrix.
    """
    
    # Convert similarity to distance
    D = 1 - S
    np.fill_diagonal(D, 0)

    # Clip negative distances to zero
    D = np.clip(D, a_min=0, a_max=None)

    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(D), method=method)

    # Apply optimal leaf ordering
    ordered_linkage_matrix = optimal_leaf_ordering(linkage_matrix, squareform(D))

    # Get the order of the leaves
    order = leaves_list(ordered_linkage_matrix)

    # Reorder the matrix
    S_reordered = S[np.ix_(order, order)]

    return S_reordered, order

def visualize_sim_matrix(A: torch.Tensor, labels: List[str], title: str = 'Similarity Matrix', order:List = None):
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


def find_top_and_worst_pairs(matrix, concepts, k=10, neutrals=False):
    """
    Computes the top k, worst k, and optionally the most neutral k pairs from a given matrix 
    (e.g., cosine similarity or covariance) and returns the results in a pandas DataFrame.
    
    Args:
        matrix (torch.Tensor): A square matrix of shape (n, n) representing the pairwise measure 
                               (e.g., cosine similarity, covariance).
        concepts (list): A list of concept names corresponding to the vectors.
        k (int): The number of top, worst, and neutral pairs to return.
        neutrals (bool): If True, also computes the most neutral pairs (smallest absolute values).

    Returns:
        pd.DataFrame: A DataFrame containing the top k, worst k, and (optionally) neutral pairs 
                      with their corresponding values.
    """
    
    # Get the indices of the upper triangle of the matrix, excluding the diagonal
    n = matrix.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1)
    
    # Extract the corresponding values from the matrix
    values = matrix[triu_indices[0], triu_indices[1]]
    
    # Sort indices to find top k and worst k scores
    sorted_indices = torch.argsort(values, descending=True)
    top_k_indices = sorted_indices[:k]  # Top k
    worst_k_indices = sorted_indices[-k:].flip(0)   # Worst k
    
    # Retrieve the top k and worst k pairs
    top_k_pairs = [(triu_indices[0][i].item(), triu_indices[1][i].item()) for i in top_k_indices]
    top_k_values = values[top_k_indices]
    
    worst_k_pairs = [(triu_indices[0][i].item(), triu_indices[1][i].item()) for i in worst_k_indices]
    worst_k_values = values[worst_k_indices]
    
    # Initialize DataFrame data
    data = {
        "Top Pairs": [f"{concepts[c1]} - {concepts[c2]}" for c1, c2 in top_k_pairs],
        "Highest Values": [f"{value:.4f}" for value in top_k_values],
        "Worst Pairs": [f"{concepts[c1]} - {concepts[c2]}" for c1, c2 in worst_k_pairs],
        "Lowest Values": [f"{value:.4f}" for value in worst_k_values],
    }
    
    # Compute neutral pairs if requested
    if neutrals:
        neutral_indices = torch.argsort(torch.abs(values))[:k]  # Smallest absolute values
        neutral_pairs = [(triu_indices[0][i].item(), triu_indices[1][i].item()) for i in neutral_indices]
        neutral_values = values[neutral_indices]
        
        # Add neutral pairs to the data
        data["Neutral Pairs"] = [f"{concepts[c1]} - {concepts[c2]}" for c1, c2 in neutral_pairs]
        data["Lowest Absolute Values"] = [f"{value:.4f}" for value in neutral_values]
    
    # Create a DataFrame to display the results
    df = pd.DataFrame(data)
    
    return df