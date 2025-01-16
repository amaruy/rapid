import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import yaml

from .dataset import WindowDataset
from .model import BiLSTMDetector
from .utils import (
    calculate_accuracy,
    calculate_anomaly_scores,
    compute_class_weights,
    find_threshold,
    should_stop_early
)

@dataclass
class TrainerConfig:
    """Configuration for model training.
    
    Attributes:
        batch_size: Size of training batches
        max_epochs: Maximum number of training epochs
        lr: Learning rate
        shuffle: Whether to shuffle training data
        window_size: Size of context window
        ntp: Number of time points to predict
        early_stopping_threshold: Threshold for early stopping
        early_stopping_patience: Patience for early stopping
        scheduler_factor: Learning rate scheduler factor
        scheduler_patience: Learning rate scheduler patience
        scheduler_threshold: Learning rate scheduler threshold
        scheduler_min_lr: Minimum learning rate
    """
    batch_size: int
    max_epochs: int
    lr: float
    shuffle: bool
    window_size: int
    ntp: int
    early_stopping_threshold: float
    early_stopping_patience: int
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_min_lr: float

class DetectorTrainer:
    """Handles training of the anomaly detection model."""
    
    def __init__(
        self,
        model: BiLSTMDetector,
        config: TrainerConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize training components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
    
    def _setup_training(
        self,
        target_size: int,
        target_labels: torch.Tensor
    ) -> None:
        """Setup training components."""
        # Setup loss with class weights
        weights = compute_class_weights(target_labels, target_size, self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            threshold=self.config.scheduler_threshold,
            min_lr=self.config.scheduler_min_lr
        )
    
    def _calculate_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        target_idx: int
    ) -> torch.Tensor:
        """Calculate loss for a batch."""
        return self.criterion(
            output.squeeze().softmax(dim=-1),
            target[:,:,target_idx].squeeze()
        )
    
    def validate(
        self,
        data_loader: DataLoader,
        target_idx: int,
        get_scores: bool = False
    ) -> Tuple[float, float, Optional[List[float]]]:
        """Validate the model."""
        self.model.eval()
        losses = []
        accuracies = []
        anomaly_scores = [] if get_scores else None
        
        for sequence, target in data_loader:
            with torch.no_grad():
                output = self.model(sequence)
                
                # Calculate metrics
                loss = self._calculate_loss(output, target, target_idx)
                losses.append(loss.item())
                
                acc = calculate_accuracy(output, target, target_idx)
                accuracies.append(acc)
                
                if get_scores:
                    scores = calculate_anomaly_scores(output, target, target_idx)
                    anomaly_scores.extend(scores.cpu().numpy())
        
        self.model.train()
        return (
            np.mean(losses),
            np.mean(accuracies),
            anomaly_scores
        )
    
    def fit(
        self,
        train_data: Dataset,
        target_idx: int,
        target_size: int,
        validate_data: Optional[Dataset] = None,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """Train the model."""
        # Create data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle
        )
        
        if validate_data:
            valid_loader = DataLoader(
                validate_data,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Get sample target labels for class weights
        sample_batch = next(iter(train_loader))
        target_labels = sample_batch[1][:,:,target_idx]
        
        # Setup training components
        self._setup_training(target_size, target_labels)
        
        # Training loop
        history = {
            'epoch': [],
            'loss': [],
            'acc': [],
            'val_epoch': [],
            'val_loss': [],
            'val_acc': []
        }
        
        threshold = None  # Initialize threshold variable
        self.model.train()
        for epoch in range(self.config.max_epochs):
            # Training epoch
            epoch_losses = []
            epoch_accuracies = []
            
            for sequence, target in train_loader:
                output = self.model(sequence)
                
                # Calculate loss and update
                loss = self._calculate_loss(output, target, target_idx)
                epoch_losses.append(loss.item())
                
                acc = calculate_accuracy(output, target, target_idx)
                epoch_accuracies.append(acc)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Record training metrics
            epoch_loss = np.mean(epoch_losses)
            epoch_acc = np.mean(epoch_accuracies)
            
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)
            history['epoch'].append(epoch)
            
            self.scheduler.step(epoch_loss)
            
            # Validation if requested
            if validate_data:
                val_loss, val_acc, anomaly_scores = self.validate(
                    valid_loader,
                    target_idx,
                    get_scores=True
                )
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_epoch'].append(epoch)
                
                # Update threshold with latest validation scores
                threshold = find_threshold(anomaly_scores)
                
                self.logger.info(
                    f'Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, '
                    f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}'
                )
            else:
                self.logger.info(
                    f'Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}'
                )
            
            # Check early stopping
            if should_stop_early(
                history['loss'],
                self.config.early_stopping_threshold,
                self.config.early_stopping_patience
            ):
                self.logger.info("Early stopping triggered")
                if validate_data:
                    self.logger.info(f'Final anomaly threshold: {threshold:.4f}')
                break
        
        # Save if requested
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_dir / "detector.pt")
            
            # Save config
            config_dict = {
                'model_state': str(save_dir / "detector.pt"),
                'threshold': threshold,  # This will be None if no validation data
                **vars(self.config)
            }
            with open(save_dir / "config.yaml", 'w') as f:
                yaml.dump(config_dict, f)
                
            self.logger.info(f"Model saved to {save_dir}")
        
        return history 