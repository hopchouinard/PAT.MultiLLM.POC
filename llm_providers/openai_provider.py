# llm_providers/openai_provider.py

import openai
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the OpenAI provider with API key from configuration.
        """
        config = get_config()
        self.api_key = config.get('openai_api_key')
        if not self.api_key:
            logger.error("OpenAI API key not found in configuration")
            raise ValueError("OpenAI API key not found in configuration")
        openai.api_key = self.api_key
        logger.info("OpenAIProvider initialized successfully")

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
        model = kwargs.get('model', 'gpt-4o-mini')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with model: {model}")
        try:
            if model.startswith('gpt-4o-mini') or model.startswith('gpt-4o'):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].text.strip()
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI API call: {str(e)}", exc_info=True)
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
        model = kwargs.get('model', 'gpt-4o-mini')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with model: {model}")
        try:
            if model.startswith('gpt-4o-mini') or model.startswith('gpt-4o'):
                stream = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.get("content"):
                        yield chunk.choices[0].delta.content
            else:
                stream = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].text:
                        yield chunk.choices[0].text
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API streaming error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI API streaming call: {str(e)}", exc_info=True)
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
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'text-embedding-ada-002')
        logger.info(f"Generating embedding with model: {model}")
        try:
            response = openai.Embedding.create(input=[text], model=model)
            return response['data'][0]['embedding']
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API embedding error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI embedding: {str(e)}", exc_info=True)
            raise

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models from OpenAI.

        Returns:
            List[Dict[str, Any]]: A list of available models and their details.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info("Fetching available models from OpenAI")
        try:
            models = openai.Model.list()
            return [{"id": model.id, "created": model.created} for model in models.data]
        except openai.error.OpenAIError as e:
            logger.error(f"Error fetching OpenAI models: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching OpenAI models: {str(e)}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the OpenAI model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'assistant') and 'content' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'gpt-4o-mini')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with model: {model}")
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage
            }
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API chat error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI API chat call: {str(e)}", exc_info=True)
            raise

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model (str): The ID of the model to retrieve information for.

        Returns:
            Dict[str, Any]: Information about the model.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info(f"Getting model info for: {model}")
        try:
            model_info = openai.Model.retrieve(model)
            return {
                "id": model_info.id,
                "created": model_info.created,
                "owned_by": model_info.owned_by
            }
        except openai.error.OpenAIError as e:
            logger.error(f"Error retrieving OpenAI model info: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving OpenAI model info: {str(e)}", exc_info=True)
            raise