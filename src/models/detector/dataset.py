from torch.utils.data import Dataset
import torch

class WindowDataset(Dataset):
    """Dataset for windowed time series data.
    
    Creates windows of sequential data for training/inference.
    Each window contains context before and after the target sequence.
    
    Attributes:
        data: Input tensor data
        window_size: Size of the context window
        ntp: Number of time points to predict
    """
    
    def __init__(self, data: torch.Tensor, window_size: int, ntp: int):
        self.data = data
        self.window_size = window_size
        self.ntp = ntp

    def __len__(self) -> int:
        """Return the number of available windows in the data."""
        return self.data.size()[0] - self.window_size*2 - self.ntp + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a window of data and its corresponding target.
        
        Args:
            index: Index of the window
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        # Get full window including context
        input = self.data[index:index + self.window_size * 2 + self.ntp]
        
        # Get target sequence from middle of window
        label = self.data[index + self.window_size:index + self.window_size + self.ntp]
        
        return input, label 