import torch
import numpy as np
from typing import Optional
from scipy.stats import rankdata

def calculate_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    target_idx: int
) -> float:
    """Calculate accuracy between predictions and targets.
    
    Args:
        output: Model output logits
        target: Target tensor
        target_idx: Index of target variable
        
    Returns:
        Accuracy value
    """
    predictions = torch.argmax(output, dim=-1)
    correct = (target[:,:,target_idx] == predictions)
    return torch.mean(correct.float()).item()

def calculate_anomaly_scores(
    output: torch.Tensor,
    target: torch.Tensor,
    target_idx: int
) -> torch.Tensor:
    """Calculate anomaly scores from model output.
    
    Args:
        output: Model output logits
        target: Target tensor
        target_idx: Index of target variable
        
    Returns:
        Tensor of anomaly scores
    """
    output_probs = output.softmax(dim=-1)
    target_probs = torch.gather(
        output_probs.squeeze(),
        1,
        target[:,:,target_idx].reshape((-1, 1))
    )
    return 1 - target_probs

def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """Compute class weights for imbalanced data.
    
    Args:
        labels: Target labels
        num_classes: Number of classes
        device: Device to place weights on
        
    Returns:
        Tensor of class weights
    """
    class_frequencies = torch.bincount(
        labels.squeeze(),
        minlength=num_classes
    )
    total_samples = torch.sum(class_frequencies)
    weights = torch.where(
        class_frequencies > 0,
        total_samples / (num_classes * class_frequencies),
        torch.tensor(0.).to(device)
    )
    return weights

def find_threshold(
    anomaly_scores: np.ndarray,
    p_value: float = 0.001
) -> float:
    """Find anomaly threshold using ECDF.
    
    Args:
        anomaly_scores: Array of anomaly scores
        p_value: P-value for threshold
        
    Returns:
        Threshold value as a float
    """
    sorted_scores = np.sort(anomaly_scores)
    n = len(sorted_scores)
    ecdf = rankdata(sorted_scores, method='average') / n
    threshold_idx = np.argmax(ecdf >= (1 - p_value))
    return float(sorted_scores[threshold_idx])

def should_stop_early(
    losses: list[float],
    threshold: float,
    patience: int
) -> bool:
    """Check if training should stop early.
    
    Args:
        losses: List of loss values
        threshold: Minimum change threshold
        patience: Number of epochs to check
        
    Returns:
        Whether to stop training
    """
    if len(losses) < patience:
        return False
        
    recent_losses = losses[-patience:]
    if max(recent_losses) - min(recent_losses) < threshold:
        return True
        
    return False 