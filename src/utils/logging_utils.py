import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    """Setup a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'{name}_{timestamp}.log'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 