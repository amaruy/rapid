import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple

from .dataset import WindowDataset
from .model import BiLSTMDetector
from .utils import calculate_accuracy, calculate_anomaly_scores

class DetectorPredictor:
    """Handles inference for the anomaly detection model.
    
    Attributes:
        model: The BiLSTM model
        device: Device to run inference on
        window_size: Size of context window
        batch_size: Batch size for inference
        threshold: Anomaly detection threshold
        logger: Logger instance
    """
    
    def __init__(
        self,
        model: BiLSTMDetector,
        device: torch.device,
        window_size: int,
        batch_size: int,
        threshold: float,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.batch_size = batch_size
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.model.eval()
    
    def predict(
        self,
        data: torch.Tensor,
        target_idx: int,
        uuids: pd.Series
    ) -> Tuple[float, Dict[str, float]]:
        """Run inference on data.
        
        Args:
            data: Input tensor data
            target_idx: Index of target variable
            uuids: Series of UUIDs for each data point
            
        Returns:
            Tuple of (accuracy, dict of UUID to anomaly score)
        """
        # Create dataset and loader
        dataset = WindowDataset(
            data=data,
            window_size=self.window_size,
            ntp=1
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Run inference
        all_scores = []
        accuracies = []
        
        with torch.no_grad():
            for sequence, target in data_loader:
                output = self.model(sequence)
                
                scores = calculate_anomaly_scores(output, target, target_idx)
                all_scores.extend(scores.cpu().numpy())
                
                acc = calculate_accuracy(output, target, target_idx)
                accuracies.append(acc)
        
        # Calculate overall accuracy
        total_acc = np.mean(accuracies)
        
        # Map scores to UUIDs
        result = {}
        for i, uid in enumerate(uuids):
            if i < self.window_size or i >= len(uuids) - self.window_size:
                result[uid] = 0.0
            else:
                result[uid] = all_scores[i - self.window_size][0]
        
        return total_acc, result 