import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve, average_precision_score
from typing import Literal, Union


def get_accuracy(y_hat, y, se=False):
    """
    Compute accuracy for multi-label binary classification.
    
    Args:
        y_hat (torch.Tensor): Predicted probabilities with shape (batch_size, n_classes), values in [0, 1].
        y (torch.Tensor): Ground truth labels with shape (batch_size, n_classes), values in {0, 1}.
        se (bool): Whether to compute the standard error (SE) of the accuracy.

    Returns:
        float: Accuracy score.
        (Optional) float: Standard error if `se` is True.
    """    
    # Threshold probabilities at 0.5 to get predicted classes
    y_pred = (y_hat >= 0.5).float()
    
    # Compute accuracy
    accuracy = (y_pred == y).float().mean().item()  # Count samples where all classes match
    
    if se:
        # Compute standard error
        se = np.sqrt(accuracy * (1 - accuracy) / y.size(0))
        return accuracy, se
    
    return accuracy


def get_f1(y_hat, y):
    """
    Compute F1 score for multi-label binary classification.
    
    Args:
        y_hat (torch.Tensor): Predicted probabilities with shape (batch_size, n_classes), values in [0, 1].
        y (torch.Tensor): Ground truth labels with shape (batch_size, n_classes), values in {0, 1}.
    
    Returns:
        float: Macro F1 score.
    """
    # Threshold probabilities at 0.5 to get predicted classes
    y_pred = (y_hat >= 0.5).int()
    
    # Convert to numpy arrays for compatibility with sklearn
    y_true_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Compute F1 scores for each class individually
    per_class_f1 = []
    for i in range(y_true_np.shape[1]):  # Loop over each class
        f1 = f1_score(y_true_np[:, i], y_pred_np[:, i], zero_division=0)
        per_class_f1.append(f1)
    
    # Compute mean F1 score across all classes
    mean_f1 = np.mean(per_class_f1)

    return mean_f1


def get_auc(y_hat, y):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for multi-label binary classification.

    Args:
        y_hat (torch.Tensor): Predicted probabilities with shape (batch_size, n_classes), values in [0, 1].
        y (torch.Tensor): Ground truth labels with shape (batch_size, n_classes), values in {0, 1}.

    Returns:
        float: The AUC-ROC score, averaged across all classes.
    """
    # Apply sigmoid to map logits to probabilities
    y_hat = y_hat.detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()

    try:
        # Compute AUC-ROC score using macro-average for multi-label
        auc_score = roc_auc_score(y_true, y_hat, average='macro')
    except ValueError:  # Handle cases where score computation fails
        auc_score = 0.0
    
    return auc_score


def get_avg_precision(y_hat, y):
    """
    Compute the Average Precision (AP) score for multi-label binary classification.

    Args:
        y_hat (torch.Tensor): Predicted probabilities with shape (batch_size, n_classes), values in [0, 1].
        y (torch.Tensor): Ground truth labels with shape (batch_size, n_classes), values in {0, 1}.

    Returns:
        float: The average precision score, averaged across all classes (macro-average).
    """
    # Apply sigmoid to map logits to probabilities
    y_hat = y_hat.detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()

    try:
        # Compute Average Precision score using macro-average for multi-label
        avg_precision = average_precision_score(y_true, y_hat, average='macro')
    except ValueError:  # Handle cases where score computation fails
        avg_precision = 0.0
    
    return avg_precision


def compute_auc_performance(
    cavs: torch.Tensor,
    activations: torch.Tensor,
    labels: torch.Tensor,
    *,
    average: Literal["micro", "macro", "weighted", "samples", None] = None,
    eps: float = 1e-12,
    ) -> Union[np.ndarray, float]:
    # cavs          (k, d)
    # activations   (batch_size, d)
    # labels        (batch_size, k)

    cavs = cavs.to('cpu')
    activations = activations.to('cpu')
    labels = labels.to('cpu')

    with torch.no_grad():
        cavs_normalized = torch.nn.functional.normalize(cavs, dim=1, eps=eps)
        activations_normalized = torch.nn.functional.normalize(activations, dim=1, eps=eps)
        scores = activations_normalized @ cavs_normalized.T       # (batch_size, k)

    y_true  = labels.cpu().numpy()
    y_score = scores.cpu().numpy()

    # Handle columns that contain only one class to avoid sklearn errors
    single_class = (y_true.sum(axis=0) == 0) | (y_true.sum(axis=0) == y_true.shape[0])

    if average is None:                      # per‑concept vector (default, back‑compat)
        aucs = np.full(y_true.shape[1], np.nan)
        if (~single_class).any():
            aucs[~single_class] = roc_auc_score(
                y_true[:, ~single_class],
                y_score[:, ~single_class],
                average=None,
            )
        return aucs
    else:                                    # any of “micro|macro|weighted|samples”
        # For micro‑/samples‑averages sklearn itself copes with single‑class columns
        return roc_auc_score(y_true, y_score, average=average)


def get_uniqueness(cavs: torch.Tensor):
    """
    Compute the uniqueness score for each Concept Activation Vector (CAV).
    
    The uniqueness score quantifies how distinct each CAV is compared to others based on cosine similarity.
    A higher score indicates that a CAV is more orthogonal (unique) relative to others.

    Args:
        cavs (torch.Tensor): Concept Activation Vectors with shape (k, d), where k is the number of concepts
                             and d is the feature dimension.

    Returns:
        np.ndarray: Uniqueness scores for each CAV as a NumPy array of shape (k,).
    """
    # Normalize cavs
    cavs_normalized = cavs / cavs.norm(p=2, dim=1, keepdim=True)

    # Cosine similarity matrix
    cos_sim_matrix = cavs_normalized @ cavs_normalized.T
    cos_sim_matrix = cos_sim_matrix.to('cpu').numpy()
    n = cos_sim_matrix.shape[0]
    
    # Compute uniqueness from cos sim matrix 
    uniqueness = 1 - np.average(np.abs(cos_sim_matrix), axis=0) + np.abs(np.diag(cos_sim_matrix)) / n

    return uniqueness

