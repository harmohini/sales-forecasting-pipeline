"""
Logging component for data ingestion and pipeline activities.
"""

import logging
import sys
from config.settings import LOG_LEVEL, LOGS_DIR

def setup_logger(name="ingestion_logger", log_file="ingestion.log"):
    """
    Configure and return a standard logger instance.
    Logs to both file (logs/ingestion.log) and standard output.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    
    # Avoid duplicate handlers if logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File handler
        file_handler = logging.FileHandler(LOGS_DIR / log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        logger.addHandler(console_handler)
        
    return logger

