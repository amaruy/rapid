from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
import logging

@dataclass
class EncoderConfig:
    """Configuration for categorical encoder.
    
    Attributes:
        min_count: Minimum frequency for a value to be encoded
        pad_value: Value used for padding
        missing_value: Value used for unknown/missing entries
    """
    min_count: int = 1
    pad_value: int = 0
    missing_value: int = 0

class Encoder:
    """Encodes categorical values to integer indices.
    
    A general-purpose encoder for converting categorical values to integer indices,
    with support for minimum frequency filtering and handling of missing values.
    
    Attributes:
        min_count: Minimum frequency for a value to be encoded
        pad_value: Value used for padding
        missing_value: Value used for unknown/missing entries
        label2code: Dictionary mapping labels to codes
        logger: Logger instance
    """
    
    def __init__(self, config: EncoderConfig, logger: Optional[logging.Logger] = None):
        self.min_count = config.min_count
        self.pad_value = config.pad_value
        self.missing_value = config.missing_value
        self.label2code: Dict[Any, int] = {np.nan: config.missing_value}
        self.logger = logger or logging.getLogger(__name__)
        
    def fit(self, vector: Union[List, np.ndarray, pd.Series]) -> None:
        """Fits the encoder to the input vector."""
        if len(vector) == 0:
            self.logger.warning("Empty vector provided for fitting")
            return
            
        value_counts = pd.Series(vector).value_counts()
        categories = value_counts[value_counts >= self.min_count].index
        
        self.logger.info(f"Found {len(categories)} categories with min_count >= {self.min_count}")
        
        current_max_code = max(self.label2code.values(), default=self.pad_value)
        new_code_start = current_max_code + 1
        
        for name in categories:
            if name not in self.label2code:
                self.label2code[name] = new_code_start
                new_code_start += 1
                
    def transform(self, vector: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Transforms labels to codes."""
        if len(vector) == 0:
            self.logger.warning("Empty vector provided for transform")
            return np.array([], dtype=int)
            
        encoded = np.array([self.label2code.get(x, self.missing_value) for x in vector], dtype=int)
        self.logger.debug(f"Transformed {len(vector)} values")
        return encoded
        
    def fit_transform(self, vector: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Fits the encoder and transforms the input data."""
        self.fit(vector)
        return self.transform(vector)
        
    def save(self, path: Union[str, Path]) -> None:
        """Saves the encoder to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.logger.info(f"Saved encoder to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Encoder':
        """Loads an encoder from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No encoder found at {path}")
            
        with open(path, 'rb') as f:
            encoder = pickle.load(f)
        return encoder
        
    def __len__(self) -> int:
        return len(self.label2code)

def save_encoders(
    encoders: Dict[str, Encoder],
    path: Union[str, Path],
    logger: Optional[logging.Logger] = None
) -> None:
    """Save multiple encoders to a single file.
    
    Args:
        encoders: Dictionary of encoder name to Encoder instance
        path: Path to save the encoders
        logger: Optional logger instance
    """
    logger = logger or logging.getLogger(__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(encoders, f)
    logger.info(f"Saved encoders to {path}")

def load_encoders(
    path: Union[str, Path],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Encoder]:
    """Load multiple encoders from a single file.
    
    Args:
        path: Path to load the encoders from
        logger: Optional logger instance
        
    Returns:
        Dictionary of encoder name to Encoder instance
    """
    logger = logger or logging.getLogger(__name__)
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"No encoders found at {path}")
        
    with open(path, 'rb') as f:
        encoders = pickle.load(f)
    logger.info(f"Loaded encoders from {path}")
    return encoders 