import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve, average_precision_score, confusion_matrix
from typing import Literal, Union, Sequence


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

def get_accuracy_mc(y_hat, y, se=False):
    if y.dim() == 2:
        accuracy = ((y_hat.sigmoid() >.5).long() == y).float().mean().item()
    else:
        accuracy = (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
    if se:
        se = np.sqrt(accuracy * (1 - accuracy) / len(y))
        return accuracy, se
    return accuracy

def get_f1_mc(y_hat, y):
    pred = (y_hat.sigmoid() >.5).long()if y.dim() == 2 else y_hat.argmax(dim=1)
    return f1_score(pred.detach().cpu(), y.detach().cpu(), average='macro')


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


def get_confusion_matrices(y_hat, y, threshold=0.5):
    """
    Compute the per-concept confusion matrix for binary classification.
    Args:
        y_hat (torch.Tensor): Predicted probabilities with shape (batch_size, n_concept), values in [0, 1].
        y (torch.Tensor): Ground truth labels with shape (batch_size, n_concept), values in {0, 1}.
    Returns:
        np.ndarray: Confusion matrix of shape (n_concepts, 2, 2) for binary classification.
    """
    n_concepts = y_hat.shape[1]

    # Threshold probabilities to get predicted classes
    y_pred = (y_hat >= threshold).int()
    y_true = y.int()

    # Convert to numpy arrays for compatibility with sklearn
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # Compute confusion matrices per concept
    cm = np.zeros((n_concepts, 2, 2), dtype=int)
    for i in range(n_concepts):
        cm[i] = confusion_matrix(y_true_np[:, i], y_pred_np[:, i], labels=[0, 1])

    return cm


def get_auconf(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    *,
    n_thresholds: int = 101
) -> np.ndarray:
    """
    Compute the per-concept confusion matrix **integrated over the decision
    threshold** in the closed interval [0,1].

    The returned matrix is analogous to AUROC: each entry (TN, FP, FN, TP)
    corresponds to the area under its curve as the threshold sweeps from 0
    to 1, i.e. the *average* count of that entry across all thresholds.

    Args
    ----
    y_hat:
        Predicted probabilities with shape ``(batch_size, n_concepts)``, values
        in [0, 1].
    y:
        Ground-truth labels with shape ``(batch_size, n_concepts)``, values in
        {0, 1}.
    n_thresholds:
        Number of evenly-spaced thresholds to sample when *thresholds* is
        *None*.

    Returns
    -------
    np.ndarray
        Integrated confusion matrix of shape ``(n_concepts, 2, 2)`` with
        floating-point dtype.
    """
    # Prepare thresholds
    thresholds = torch.linspace(0.0, 1.0, steps=n_thresholds, device=y_hat.device, dtype=y_hat.dtype)
    n_thr = thresholds.numel()
    n_concepts = y_hat.shape[1]
    cms = np.zeros((n_thr, n_concepts, 2, 2), dtype=np.float64)

    # Compute confusion matrix for every threshold
    y_true_np = y.int().cpu().numpy()
    for t_idx, thr in enumerate(thresholds):
        y_pred_np = (y_hat >= thr).int().cpu().numpy()

        for i in range(n_concepts):
            cms[t_idx, i] = confusion_matrix(
                y_true_np[:, i], y_pred_np[:, i], labels=[0, 1]
            )
    
    # Integrate (trapezoidal rule) over the threshold axis
    thr_np = thresholds.cpu().numpy()
    cm_integral = np.trapz(cms, thr_np, axis=0)  # area under each entry‑curve

    return cm_integral


def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return accuracy, false_positive_rate, false_negative_rate, recall, precision


def get_auc_label(y_true, model_outs, label):
    if y_true.dim() == 2:
        fpr, tpr, _ = roc_curve(y_true.numpy()[:,label], model_outs.numpy()[:,label])
    else:
        fpr, tpr, _ = roc_curve(y_true.numpy(), model_outs.numpy()[:, label], pos_label=label)
    return auc(fpr, tpr)

def get_fpr_label(y_true, model_preds, label):
    cm = confusion_matrix(y_true[:,label], model_preds[:,label], labels=(0,1))
    fp = cm[0, 1]
    tn = cm[0, 0]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr

def get_fnr_label(y_true, model_preds, label):
    cm = confusion_matrix(y_true[:, label], model_preds[:, label], labels=(0, 1))
    fn = cm[1, 0]  # False Negatives
    tp = cm[1, 1]  # True Positives
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return fnr