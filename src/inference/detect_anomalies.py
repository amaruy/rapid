from pathlib import Path
import yaml
import torch
import pandas as pd
import pickle
import logging
from typing import Dict, Tuple

from src.utils.logging_utils import setup_logger
from src.utils.encoder import load_encoders
from src.models.detector.model import BiLSTMDetector
from src.models.detector.predictor import DetectorPredictor

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def encode_data(
    df: pd.DataFrame,
    encoders: dict,
    device: torch.device
) -> torch.Tensor:
    """Encode dataframe columns to tensor.
    
    Args:
        df: Input DataFrame
        encoders: Dictionary of encoders
        device: Torch device
        
    Returns:
        Encoded tensor data
    """
    # Encode each column
    process = torch.from_numpy(encoders['process'].transform(df['processName'])).type(torch.long)
    event = torch.from_numpy(encoders['event'].transform(df['event'])).type(torch.long)
    obj_type = torch.from_numpy(encoders['object_type'].transform(df['objectType'])).type(torch.long)
    obj_data = torch.from_numpy(encoders['object_data'].transform(df['objectData'])).type(torch.long)
    
    # Stack and move to device
    return torch.stack((process, event, obj_type, obj_data)).to(device).T

def setup_model(
    config: dict,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[BiLSTMDetector, dict]:
    """Setup the BiLSTM model.
    
    Args:
        config: Model configuration
        device: Torch device
        logger: Logger instance
        
    Returns:
        Tuple of (initialized model, encoders dictionary)
    """
    # Load encoders to get input sizes
    encoders = load_encoders(config['encoders_path'], logger)
    
    # Setup input sizes
    input_sizes = {
        'subject': len(encoders['process']),
        'event': len(encoders['event']),
        'objectType': len(encoders['object_type']),
        'objectData': len(encoders['object_data'])
    }
    
    # Load model config
    model_config = load_config(Path('config/model_config.yaml'))
    
    # Initialize model
    model = BiLSTMDetector(
        input_sizes=input_sizes,
        embedding_dims=model_config['model']['embedding_dims'],
        hidden_size=model_config['model']['architecture']['hidden_size'],
        num_layers=model_config['model']['architecture']['num_layers'],
        fc1_size=model_config['model']['architecture']['fc1_size'],
        fc2_size=len(encoders['event']),  # Output size is number of event types
        window_size=config['window_size'],
        object_weights_path=config['embedder_path'],
        logger=logger
    ).to(device)
    
    # Load model state
    state_dict = torch.load(config['model_state_path'], map_location=device)
    model.load_state_dict(state_dict)
    
    return model, encoders

def detect_anomalies(
    data_path: Path,
    config: dict,
    logger: logging.Logger
) -> Tuple[float, Dict[str, float]]:
    """Run anomaly detection on data.
    
    Args:
        data_path: Path to input data
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (accuracy, anomaly scores dict)
    """
    # Setup device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_pickle(data_path)
    
    # Filter events if specified
    if config.get('exclude_events'):
        df = df[~df['event'].isin(config['exclude_events'])]
    
    # Setup model
    logger.info("Setting up model")
    model, encoders = setup_model(config, device, logger)
    
    # Encode data
    logger.info("Encoding data")
    encoded_data = encode_data(df, encoders, device)
    
    # Setup predictor
    predictor = DetectorPredictor(
        model=model,
        device=device,
        window_size=config['window_size'],
        batch_size=config['batch_size'],
        threshold=config['threshold'],
        logger=logger
    )
    
    # Run prediction
    logger.info("Running anomaly detection")
    accuracy, anomaly_scores = predictor.predict(
        data=encoded_data,
        target_idx=1,  # Index for event prediction
        uuids=df['uuid']
    )
    
    return accuracy, anomaly_scores

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    
    # Load configs
    base_config = load_config(root_dir / "config" / "config.yaml")
    infer_config = load_config(root_dir / "config" / "inference_config.yaml")
    
    # Setup logging
    logger = setup_logger(
        "anomaly_detection",
        root_dir / base_config['paths']['logs_dir']
    )
    logger.info("Starting anomaly detection")
    
    # Update paths in config to be absolute
    config = infer_config['anomaly_detection']
    for key in ['model_state_path', 'embedder_path', 'encoders_path', 'output_dir']:
        config[key] = str(root_dir / config[key])
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run anomaly detection
    data_path = root_dir / base_config['paths']['data_dir'] / "test_logs.pkl"
    accuracy, anomaly_scores = detect_anomalies(data_path, config, logger)
    
    # Log results
    logger.info(f"Anomaly detection accuracy: {accuracy:.4f}")
    logger.info(f"Found {len(anomaly_scores)} potential anomalies")
    
    # Save results
    output_path = output_dir / config['scores_filename']
    with open(output_path, 'wb') as f:
        pickle.dump(anomaly_scores, f)
    logger.info(f"Saved anomaly scores to {output_path}")

if __name__ == "__main__":
    main() 