# llm_providers/__init__.py

from typing import Dict, Type
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .lmstudio_provider import LMStudioProvider
from utils.logging import logger

# Dictionary mapping provider names to their respective classes
PROVIDER_MAP: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider
}

def get_provider(provider_name: str) -> LLMProvider:
    """
    Factory function to create and return an instance of the specified LLM provider.

    Args:
        provider_name (str): The name of the provider to instantiate.

    Returns:
        LLMProvider: An instance of the specified provider.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    provider_name = provider_name.lower()
    if provider_name not in PROVIDER_MAP:
        logger.error(f"Unsupported provider: {provider_name}")
        raise ValueError(f"Unsupported provider: {provider_name}")
    
    provider_class = PROVIDER_MAP[provider_name]
    logger.info(f"Creating instance of {provider_name} provider")
    return provider_class()

def list_supported_providers() -> list[str]:
    """
    Returns a list of all supported provider names.

    Returns:
        list[str]: A list of supported provider names.
    """
    return list(PROVIDER_MAP.keys())

# Version of the llm_providers package
__version__ = "1.0.0"

# Export the LLMProvider base class and all provider classes
__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "get_provider",
    "list_supported_providers"
]