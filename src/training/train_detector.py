from pathlib import Path
import yaml
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import setup_logger
from src.utils.encoder import load_encoders
from src.models.detector.model import BiLSTMDetector
from src.models.detector.dataset import WindowDataset
from src.models.detector.trainer import DetectorTrainer, TrainerConfig

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def encode_data(
    df: pd.DataFrame,
    encoders: dict,
    device: torch.device
) -> torch.Tensor:
    """Encode dataframe columns to tensor."""
    # Encode each column
    process = torch.from_numpy(encoders['process'].transform(df['processName'])).type(torch.long)
    event = torch.from_numpy(encoders['event'].transform(df['event'])).type(torch.long)
    obj_type = torch.from_numpy(encoders['object_type'].transform(df['objectType'])).type(torch.long)
    obj_data = torch.from_numpy(encoders['object_data'].transform(df['objectData'])).type(torch.long)
    
    # Stack and move to device
    return torch.stack((process, event, obj_type, obj_data)).to(device).T

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    
    # Load configs
    base_config = load_config(root_dir / "config" / "config.yaml")
    model_config = load_config(root_dir / "config" / "model_config.yaml")
    
    # Setup logging
    logger = setup_logger(
        "detector_training",
        root_dir / base_config['paths']['logs_dir']
    )
    logger.info("Starting detector training")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    data_path = root_dir / base_config['paths']['data_dir'] / "train_logs.pkl"
    logger.info(f"Loading training data from {data_path}")
    train_data = pd.read_pickle(data_path)
    logger.info(f"Loaded data with shape {train_data.shape}")
    
    # Load encoders
    encoders_path = root_dir / base_config['paths']['artifacts_dir'] / "encoders" / "encoders.pkl"
    logger.info(f"Loading encoders from {encoders_path}")
    encoders = load_encoders(encoders_path, logger)
    
    # Setup input sizes
    input_sizes = {
        'subject': len(encoders['process']),
        'event': len(encoders['event']),
        'objectType': len(encoders['object_type']),
        'objectData': len(encoders['object_data'])
    }
    logger.info(f"Input sizes: {input_sizes}")
    
    # Filter data if needed
    if model_config['data'].get('min_proc_size'):
        min_size = model_config['data']['min_proc_size']
        proc_counts = train_data['processUUID'].value_counts()
        valid_procs = proc_counts[proc_counts >= min_size].index
        train_data = train_data[train_data['processUUID'].isin(valid_procs)]
        logger.info(f"Filtered data to shape {train_data.shape}")
    
    # Encode data
    logger.info("Encoding data")
    encoded_data = encode_data(train_data, encoders, device)
    
    # Split data
    train_data, valid_data = train_test_split(
        encoded_data,
        test_size=0.1,
        shuffle=False
    )
    
    # Create datasets
    train_dataset = WindowDataset(
        data=train_data,
        window_size=model_config['training']['window_size'],
        ntp=model_config['training']['ntp']
    )
    valid_dataset = WindowDataset(
        data=valid_data,
        window_size=model_config['training']['window_size'],
        ntp=model_config['training']['ntp']
    )
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    
    # Initialize model
    model = BiLSTMDetector(
        input_sizes=input_sizes,
        embedding_dims=model_config['model']['embedding_dims'],
        hidden_size=model_config['model']['architecture']['hidden_size'],
        num_layers=model_config['model']['architecture']['num_layers'],
        fc1_size=model_config['model']['architecture']['fc1_size'],
        fc2_size=len(encoders['event']),  # Output size is number of event types
        window_size=model_config['training']['window_size'],
        object_weights_path=root_dir / base_config['paths']['artifacts_dir'] / "models" / "embeddings" / "embedder.pt",
        logger=logger
    ).to(device)
    
    # Setup trainer config
    trainer_config = TrainerConfig(
        batch_size=model_config['training']['batch_size'],
        max_epochs=model_config['training']['max_epochs'],
        lr=model_config['training']['lr'],
        shuffle=model_config['training']['shuffle'],
        window_size=model_config['training']['window_size'],
        ntp=model_config['training']['ntp'],
        early_stopping_threshold=model_config['training']['early_stopping']['threshold'],
        early_stopping_patience=model_config['training']['early_stopping']['patience'],
        scheduler_factor=model_config['training']['scheduler']['factor'],
        scheduler_patience=model_config['training']['scheduler']['patience'],
        scheduler_threshold=model_config['training']['scheduler']['threshold'],
        scheduler_min_lr=model_config['training']['scheduler']['min_lr']
    )
    
    # Initialize trainer
    trainer = DetectorTrainer(
        model=model,
        config=trainer_config,
        device=device,
        logger=logger
    )
    
    # Train model
    logger.info("Starting model training")
    history = trainer.fit(
        train_data=train_dataset,
        target_idx=model_config['data']['target'],
        target_size=len(encoders['event']),
        validate_data=valid_dataset,
        save_dir=root_dir / base_config['paths']['artifacts_dir'] / "models" / "detector"
    )
    
    logger.info("Training complete")

if __name__ == "__main__":
    main() 