# llm_providers/ollama_provider.py

import ollama
import requests
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class OllamaProvider(LLMProvider):
    """
    Ollama API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the Ollama provider with the host from configuration.
        """
        config = get_config()
        self.host = config.get('ollama_host', 'http://localhost:11434')
        ollama.set_host(self.host)
        logger.info(f"OllamaProvider initialized with host: {self.host}")

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
        model = kwargs.get('model', 'llama3')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with model: {model}")
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            return response['response'].strip()
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {str(e)}", exc_info=True)
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
        model = kwargs.get('model', 'llama3')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with model: {model}")
        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                },
                stream=True
            )
            for chunk in stream:
                if chunk['response']:
                    yield chunk['response']
        except requests.RequestException as e:
            logger.error(f"Ollama API streaming error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API streaming call: {str(e)}", exc_info=True)
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
        model = kwargs.get('model', 'mxbai-embed-large')
        
        logger.info(f"Generating embedding with model: {model}")
        try:
            response = ollama.embeddings(model=model, prompt=text)
            return response['embedding']
        except requests.RequestException as e:
            logger.error(f"Ollama embedding error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama embedding: {str(e)}", exc_info=True)
            raise

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models from Ollama.

        Returns:
            List[Dict[str, Any]]: A list of available models and their details.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info("Fetching available models from Ollama")
        try:
            models = ollama.list()
            return [{"name": model['name'], "modified_at": model['modified_at']} for model in models['models']]
        except requests.RequestException as e:
            logger.error(f"Error fetching Ollama models: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching Ollama models: {str(e)}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the Ollama model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'assistant') and 'content' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'llama3')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with model: {model}")
        try:
            # Convert messages to Ollama's expected format
            ollama_messages = [{"role": msg['role'], "content": msg['content']} for msg in messages]
            
            response = ollama.chat(
                model=model,
                messages=ollama_messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            
            return {
                "role": "assistant",
                "content": response['message']['content'],
                "model": model,
            }
        except requests.RequestException as e:
            logger.error(f"Ollama API chat error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API chat call: {str(e)}", exc_info=True)
            raise

    def pull_model(self, model_name: str) -> None:
        """
        Pull a model from Ollama's model library.

        Args:
            model_name (str): The name of the model to pull.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info(f"Pulling model: {model_name}")
        try:
            ollama.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
        except requests.RequestException as e:
            logger.error(f"Error pulling Ollama model: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error pulling Ollama model: {str(e)}", exc_info=True)
            raise

    def delete_model(self, model_name: str) -> None:
        """
        Delete a model from the local Ollama instance.

        Args:
            model_name (str): The name of the model to delete.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info(f"Deleting model: {model_name}")
        try:
            ollama.delete(model_name)
            logger.info(f"Successfully deleted model: {model_name}")
        except requests.RequestException as e:
            logger.error(f"Error deleting Ollama model: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting Ollama model: {str(e)}", exc_info=True)
            raise