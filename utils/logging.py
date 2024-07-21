# utils/logging.py

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
from utils.config import get_config

def setup_logger(name: str = 'llm_api_manager', log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name (str): The name of the logger. Defaults to 'llm_api_manager'.
        log_file (Optional[str]): The path to the log file. If None, logs will only be printed to console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    config = get_config()
    log_level = config.get('log_level', 'INFO')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If log file is specified, create file handler
    if log_file:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler and set level to debug
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, keep 5 old files
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Create a default logger
logger = setup_logger(log_file='logs/llm_api_manager.log')

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name. If the logger doesn't exist, it will be created.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    if name not in logging.Logger.manager.loggerDict:
        return setup_logger(name, log_file=f'logs/{name}.log')
    return logging.getLogger(name)

def set_log_level(level: str):
    """
    Set the log level for the default logger and all its handlers.

    Args:
        level (str): The log level to set. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)

    config = get_config()
    config.set('log_level', level)
    logger.info(f"Log level set to {level}")

# Example usage
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Change log level
    set_log_level('DEBUG')
    logger.debug("This debug message should now appear")

    # Get a named logger
    custom_logger = get_logger('custom_module')
    custom_logger.info("This is a message from the custom logger")