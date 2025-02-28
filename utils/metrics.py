import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve, average_precision_score


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


def compute_auc_performance(cavs, activations, labels):
    """
    Compute the AUC-ROC scores for each concept based on cosine similarity between CAVs and activations.

    Args:
        cavs (torch.Tensor): Concept Activation Vectors with shape (k, d), where k is the number of concepts
                             and d is the feature dimension.
        activations (torch.Tensor): Model activations with shape (batch_size, d).
        labels (torch.Tensor): Ground truth labels for the concepts with shape (batch_size, k), where k is the
                               number of concepts.

    Returns:
        np.ndarray: AUC-ROC scores for each concept as a NumPy array of shape (k,).
    """
    # cavs          (k, d)
    # activations   (batch_size, d)
    # labels        (batch_size, k)

    # Move to cpu
    cavs = cavs.to('cpu')
    activations = activations.to('cpu')
    labels = labels.to('cpu')

    # Normalize the CAVs and activations for cosine similarity
    cavs_normalized = cavs / cavs.norm(dim=1, keepdim=True)  # Shape: (k, d)
    activations_normalized = activations / activations.norm(dim=1, keepdim=True)  # Shape: (batch_size, d)
    
    auc_scores = []
    
    # Iterate over each concept
    for concept_idx in range(cavs.shape[0]):
        # Compute cosine similarity for the current concept
        cosine_sim = activations_normalized @ cavs_normalized[concept_idx]  # Shape: (batch_size,)
        
        # Get the true labels for the current concept
        true_labels = labels[:, concept_idx].cpu().numpy()  # Shape: (batch_size,)
        
        # Compute AUC for the current concept
        auc = roc_auc_score(true_labels, cosine_sim.cpu().numpy())
        auc_scores.append(auc)
    
    return np.array(auc_scores)

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