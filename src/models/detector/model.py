import torch
import torch.nn as nn
from typing import Dict, Optional
import logging
from pathlib import Path

def init_weights(m: nn.Module) -> None:
    """Initialize model weights.
    
    Args:
        m: Module to initialize
    """
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)

class ObjectEmbed(nn.Module):
    """Neural network module for object embeddings.
    
    Loads pretrained embeddings and optionally reduces their dimension.
    
    Attributes:
        embed_layer: Embedding layer with pretrained weights
        embedding_reducer: Linear layer for dimension reduction
    """
    
    def __init__(self, weights_path: Path, out_dim: int = 32) -> None:
        super().__init__()
        
        # Load pretrained embeddings
        state_dict = torch.load(weights_path)
        embed_weights = state_dict['embedding.weight']
        num_embeddings, embedding_dim = embed_weights.shape
        
        # Setup layers
        self.embed_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embed_layer.weight.data = embed_weights.data
        self.embed_layer.weight.requires_grad = False  # freeze embeddings
        
        self.embedding_reducer = nn.Linear(embedding_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_layer(x)
        x = self.embedding_reducer(x)
        return x

class EmbedData(nn.Module):
    """Combines multiple embeddings for different features.
    
    Attributes:
        subject_embedding: Embedding layer for process names
        event_embedding: Embedding layer for events
        objectType_embedding: Embedding layer for object types
        objectData_embedding: Pretrained embeddings for object data
    """
    
    def __init__(
        self,
        input_sizes: Dict[str, int],
        embedding_dims: Dict[str, int],
        object_weights_path: Path
    ) -> None:
        super().__init__()
        
        self.subject_embedding = nn.Embedding(
            input_sizes['subject'],
            embedding_dims['subject']
        )
        self.event_embedding = nn.Embedding(
            input_sizes['event'],
            embedding_dims['event']
        )
        self.objectType_embedding = nn.Embedding(
            input_sizes['objectType'],
            embedding_dims['objectType']
        )
        self.objectData_embedding = ObjectEmbed(
            object_weights_path,
            embedding_dims['objectData']
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Embed each feature
        subject_embedded = self.subject_embedding(input[:, :, 0])
        event_embedded = self.event_embedding(input[:, :, 1])
        objectType_embedded = self.objectType_embedding(input[:, :, 2])
        objectData_embedded = self.objectData_embedding(input[:, :, 3])
        
        # Concatenate all embeddings
        x = torch.cat([
            subject_embedded,
            event_embedded,
            objectType_embedded,
            objectData_embedded
        ], dim=-1)
        
        return x

class BiLSTMDetector(nn.Module):
    """Bidirectional LSTM for anomaly detection.
    
    Attributes:
        embedding: Combined embedding layers
        lstm: Bidirectional LSTM layers
        fc1: First fully connected layer
        fc2: Output layer
    """
    
    def __init__(
        self,
        input_sizes: Dict[str, int],
        embedding_dims: Dict[str, int],
        hidden_size: int,
        num_layers: int,
        fc1_size: int,
        fc2_size: int,
        window_size: int,
        object_weights_path: Path,
        logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup layers
        self.embedding = EmbedData(
            input_sizes,
            embedding_dims,
            object_weights_path
        )
        
        embed_dim = sum(embedding_dims.values())
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Compact weights for memory efficiency
        self.lstm.flatten_parameters()
        
        self.fc1 = nn.Linear(hidden_size*2, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Embed input
        x = self.embedding(input)
        
        # Initialize LSTM hidden states
        h0 = torch.zeros(
            self.num_layers*2,
            x.size(0),
            self.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers*2,
            x.size(0),
            self.hidden_size,
            device=x.device
        )
        
        # Process sequence
        x, _ = self.lstm(x, (h0, c0))
        
        # Apply fully connected layers
        x = self.fc1(x[:, self.window_size:-self.window_size,:])
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x 