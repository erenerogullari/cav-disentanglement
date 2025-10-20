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