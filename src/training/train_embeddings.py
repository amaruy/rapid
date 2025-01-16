from pathlib import Path
import yaml
import pandas as pd
import logging
from typing import Optional
import pickle

from src.utils.logging_utils import setup_logger
from src.utils.encoder import Encoder, EncoderConfig, save_encoders
from src.models.embeddings.models import EmbeddingConfig
from src.models.embeddings.trainer import Word2VecConfig, EmbeddingTrainer

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_encoders(
    data: pd.DataFrame,
    encoder_config: EncoderConfig,
    save_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> dict[str, Encoder]:
    """Setup or load encoders for different columns."""
    logger = logger or logging.getLogger(__name__)
    
    if save_dir and (save_dir / "encoders.pkl").exists():
        logger.info("Loading existing encoders")
        with open(save_dir / "encoders.pkl", 'rb') as f:
            return pickle.load(f)
    
    logger.info("Creating new encoders")
    encoders = {
        'process': Encoder(encoder_config, logger),
        'event': Encoder(encoder_config, logger),
        'object_type': Encoder(encoder_config, logger),
        'object_data': Encoder(encoder_config, logger)
    }
    
    # Fit encoders
    encoders['process'].fit(data['processName'])
    encoders['event'].fit(data['event'])
    encoders['object_type'].fit(data['objectType'])
    encoders['object_data'].fit(data['objectData'])
    
    logger.info(
        f"Encoder sizes - Process: {len(encoders['process'])}, "
        f"Event: {len(encoders['event'])}, "
        f"Object Type: {len(encoders['object_type'])}, "
        f"Object Data: {len(encoders['object_data'])}"
    )
    
    # Save if requested
    if save_dir:
        save_encoders(encoders, save_dir / "encoders.pkl", logger)
    
    return encoders

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    
    # Load both configs
    base_config = load_config(root_dir / "config" / "config.yaml")
    embed_config = load_config(root_dir / "config" / "embedding_config.yaml")
    
    # Setup logging
    logger = setup_logger(
        "embedding_training",
        root_dir / base_config['paths']['logs_dir']
    )
    logger.info("Starting embedding training")
    
    # Load data
    data_path = root_dir / base_config['paths']['data_dir'] / "train_logs.pkl"
    logger.info(f"Loading training data from {data_path}")
    train_data = pd.read_pickle(data_path)
    logger.info(f"Loaded data with shape {train_data.shape}")
    
    # Setup directories using base_config for paths
    artifacts_dir = root_dir / base_config['paths']['artifacts_dir']
    
    # Initialize configs using embed_config for model settings
    encoder_config = EncoderConfig(**embed_config['encoder'])
    
    # Split embedding config into Word2Vec and Embedding configs
    w2v_params = {k: v for k, v in embed_config['embedding'].items() 
                 if k not in ['reduced_dim']}
    w2v_config = Word2VecConfig(**w2v_params)
    
    embed_config = EmbeddingConfig(
        vector_size=w2v_config.vector_size,
        reduced_dim=embed_config['embedding']['reduced_dim']
    )
    
    # Setup encoders
    encoders = setup_encoders(
        train_data,
        encoder_config,
        save_dir=artifacts_dir / "encoders",
        logger=logger
    )
    
    # Initialize trainer
    trainer = EmbeddingTrainer(
        w2v_config=w2v_config,
        embed_config=embed_config,
        logger=logger
    )
    
    # Train embeddings
    logger.info("Starting embedding training")
    w2v_model, embedder = trainer.train(
        data=train_data,
        encoder=encoders['object_data'],
        text_column='objectData',
        group_columns=['processUUID', 'processName'],
        save_dir=artifacts_dir / "models" / "embeddings"
    )
    
    logger.info("Embedding training complete")

if __name__ == "__main__":
    main() 