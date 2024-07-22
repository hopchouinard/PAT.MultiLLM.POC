import anthropic
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the Anthropic provider with API key from configuration.
        """
        config = get_config()
        self.api_key = config.get('anthropic_api_key')
        if not self.api_key:
            logger.error("Anthropic API key not found in configuration")
            raise ValueError("Anthropic API key not found in configuration")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info("AnthropicProvider initialized successfully")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional arguments for the API call.

        Returns:
            str: The generated text.

        Raises:
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'claude-3-haiku-20240307')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with model: {model}")
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
                temperature=temperature
            )
            return response.completion
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Anthropic API call: {str(e)}", exc_info=True)
            raise

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate text in a streaming fashion based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional arguments for the API call.

        Yields:
            str: Chunks of generated text.

        Raises:
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'claude-3-haiku-20240307')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with model: {model}")
        try:
            with self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                stream=True
            ) as stream:
                for completion in stream:
                    yield completion.completion
        except anthropic.APIError as e:
            logger.error(f"Anthropic API streaming error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Anthropic API streaming call: {str(e)}", exc_info=True)
            raise

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text.

        Args:
            text (str): The input text to embed.
            **kwargs: Additional arguments for the API call.

        Returns:
            List[float]: The embedding vector.

        Raises:
            NotImplementedError: Anthropic doesn't currently provide a public embedding API.
        """
        logger.warning("Embedding not currently supported for Anthropic")
        raise NotImplementedError("Embedding not currently supported for Anthropic")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the Anthropic model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'assistant') and 'content' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'claude-3-haiku-20240307')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with model: {model}")
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages]
            )
            return {
                "role": "assistant",
                "content": response.content[0].text,
                "model": response.model,
                "usage": response.usage
            }
        except anthropic.APIError as e:
            logger.error(f"Anthropic API chat error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Anthropic API chat call: {str(e)}", exc_info=True)
            raise

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models from Anthropic.

        Returns:
            List[Dict[str, Any]]: A list of available models and their details.

        Note:
            As of my last update, Anthropic doesn't provide a public API to list models.
            This method returns a hardcoded list of known models.
        """
        logger.info("Fetching available models from Anthropic")
        # As Anthropic doesn't provide a public API to list models,
        # we're returning a hardcoded list of known models.
        models = [
            {"id": "claude-3-5-sonnet-20240620", "description": "Most advanced model, outperforming its predecessors, with a perfect blend of intelligence and speed."},
            {"id": "claude-3-opus-20240229", "description": "Most capable model for complex tasks"},
            {"id": "claude-3-sonnet-20240229", "description": "Ideal balance of intelligence and speed"},
            {"id": "claude-3-haiku-20240307", "description": "Fastest model, best for simple tasks and back-and-forth"},
            {"id": "claude-2.1", "description": "Powerful model for a wide range of tasks"},
            {"id": "claude-instant-1.2", "description": "Faster and more compact model for simpler tasks"}
        ]
        return models