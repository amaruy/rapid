from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

@dataclass
class EmbeddingConfig:
    """Configuration for word embeddings model.
    
    Attributes:
        vector_size: Dimension of the word vectors
        reduced_dim: Dimension to reduce embeddings to
        freeze_embeddings: Whether to freeze the embedding weights
    """
    vector_size: int = 100
    reduced_dim: int = 32
    freeze_embeddings: bool = True

class WordEmbeddings(nn.Module):
    """Neural network module for word embeddings with dimension reduction.
    
    This module takes pretrained word embeddings and adds a linear layer
    to reduce their dimensionality.
    
    Attributes:
        embedding: The embedding layer
        embedding_reducer: Linear layer for dimension reduction
        config: Model configuration
        logger: Logger instance
    """
    
    def __init__(
        self, 
        num_embeddings: int,
        embedding_matrix: np.ndarray,
        config: EmbeddingConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate input dimensions
        if embedding_matrix.shape != (num_embeddings, config.vector_size):
            raise ValueError(
                f"Embedding matrix shape {embedding_matrix.shape} doesn't match "
                f"expected shape ({num_embeddings}, {config.vector_size})"
            )
        
        # Initialize layers
        self.embedding = nn.Embedding(num_embeddings, config.vector_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = not config.freeze_embeddings
        
        self.embedding_reducer = nn.Linear(config.vector_size, config.reduced_dim)
        
        self.logger.info(
            f"Initialized WordEmbeddings with {num_embeddings} embeddings, "
            f"reducing from {config.vector_size} to {config.reduced_dim} dimensions"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of token indices
            
        Returns:
            Tensor of reduced-dimension embeddings
        """
        embedded = self.embedding(x)
        reduced = self.embedding_reducer(embedded)
        return reduced
    
    def save(self, path: Path) -> None:
        """Save model weights to disk.
        
        Args:
            path: Path to save the model weights
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        self.logger.info(f"Saved model weights to {path}")
    
    @classmethod
    def load(
        cls,
        path: Path,
        num_embeddings: int,
        embedding_matrix: np.ndarray,
        config: EmbeddingConfig,
        logger: Optional[logging.Logger] = None
    ) -> 'WordEmbeddings':
        """Load model weights from disk.
        
        Args:
            path: Path to load the model weights from
            num_embeddings: Number of embeddings
            embedding_matrix: Initial embedding matrix
            config: Model configuration
            logger: Optional logger instance
            
        Returns:
            WordEmbeddings instance with loaded weights
        """
        model = cls(num_embeddings, embedding_matrix, config, logger)
        model.load_state_dict(torch.load(path))
        return model 