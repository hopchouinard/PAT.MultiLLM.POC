# utils/__init__.py

from .config import get_config, Config
from .logging import logger, get_logger, set_log_level, setup_logger

# Version of the utils package
__version__ = "1.0.0"

# Export the main utility functions and classes
__all__ = [
    "get_config",
    "Config",
    "logger",
    "get_logger",
    "set_log_level",
    "setup_logger"
]

# Initialization code (if any)
def initialize_utils():
    """
    Perform any necessary initialization for the utils package.
    This function can be called when the package is first imported.
    """
    # For now, we'll just log that the utils package has been initialized
    logger.info("Utils package initialized")

# Call initialize_utils when the package is imported
initialize_utils()