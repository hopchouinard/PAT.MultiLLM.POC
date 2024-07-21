# utils/config.py

import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from utils.logging import logger

class Config:
    """
    Configuration management class for the LLM API Manager.
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """
        Load configuration from various sources.
        Priority: Environment variables > .env file > config.json
        """
        # Load from .env file
        load_dotenv()

        # Load from config.json if it exists
        config_file = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self._config.update(json.load(f))
                logger.info("Loaded configuration from config.json")

        # Load from environment variables, overriding existing values
        env_config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'ollama_host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
            'lmstudio_api_base': os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        }
        
        # Update config, removing None values
        self._config.update({k: v for k, v in env_config.items() if v is not None})
        logger.info("Loaded configuration from environment variables")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key (str): The configuration key to retrieve.
            default (Any, optional): The default value to return if the key is not found.

        Returns:
            Any: The configuration value, or the default if not found.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set a configuration value.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to set for the key.
        """
        self._config[key] = value
        logger.info(f"Set configuration: {key}")

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dict[str, Any]: A dictionary of all configuration key-value pairs.
        """
        return self._config.copy()

    def save_to_file(self, filename: str = 'config.json'):
        """
        Save the current configuration to a JSON file.

        Args:
            filename (str, optional): The name of the file to save to. Defaults to 'config.json'.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', filename)
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=4)
        logger.info(f"Saved configuration to {filename}")

# Global configuration instance
_config_instance = None

def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config: The global configuration instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

# Example usage
if __name__ == "__main__":
    config = get_config()
    print("OpenAI API Key:", config.get("openai_api_key"))
    print("Log Level:", config.get("log_level"))
    
    # Set a new configuration value
    config.set("new_setting", "new_value")
    print("New Setting:", config.get("new_setting"))
    
    # Print all configuration
    print("All Config:", config.get_all())
    
    # Save configuration to file
    config.save_to_file()