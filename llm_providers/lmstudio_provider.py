# llm_providers/lmstudio_provider.py

import openai
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class LMStudioProvider(LLMProvider):
    """
    LM Studio API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the LM Studio provider with API base URL from configuration.
        """
        config = get_config()
        self.api_base = config.get('lmstudio_api_base', 'http://localhost:1234/v1')
        if not self.api_base:
            logger.error("LM Studio API base not found in configuration")
            raise ValueError("LM Studio API base not found in configuration")
        self.client = openai.OpenAI(base_url=self.api_base, api_key="not-needed")
        logger.info(f"LMStudioProvider initialized with API base: {self.api_base}")

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
        model = kwargs.get('model', 'lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as e:
            logger.error(f"LM Studio API error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio API call: {str(e)}", exc_info=True)
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
        model = kwargs.get('model', 'lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with model: {model}")
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as e:
            logger.error(f"LM Studio API streaming error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio API streaming call: {str(e)}", exc_info=True)
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
        model = kwargs.get('model', 'nomic-embed-text-v1.5')
        
        logger.info(f"Generating embedding with model: {model}")
        try:
            response = self.client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except openai.OpenAIError as e:
            logger.error(f"LM Studio embedding error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio embedding: {str(e)}", exc_info=True)
            raise

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models from LM Studio.

        Returns:
            List[Dict[str, Any]]: A list of available models and their details.

        Note:
            LM Studio typically uses local models, so this method returns a predefined list.
        """
        logger.info("Fetching available models from LM Studio")
        # LM Studio typically uses local models, so we return a predefined list
        models = [
            {"id": "local-model", "name": "Default Local Model"},
            {"id": "local-embedding-model", "name": "Default Local Embedding Model"}
        ]
        return models

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the LM Studio model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'assistant') and 'content' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        model = kwargs.get('model', 'lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "model": model,
            }
        except openai.OpenAIError as e:
            logger.error(f"LM Studio API chat error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio API chat call: {str(e)}", exc_info=True)
            raise

    def set_model_parameters(self, **kwargs) -> None:
        """
        Set model parameters for LM Studio.

        Args:
            **kwargs: Model parameters to set.

        Note:
            This method doesn't actually change any settings on the LM Studio server.
            It's meant to be used as a placeholder for potential future functionality.
        """
        logger.info("Setting model parameters for LM Studio")
        # LM Studio doesn't provide an API to set model parameters directly
        # This method is a placeholder for potential future functionality
        logger.warning("Setting model parameters is not supported in LM Studio API")
        pass

    def get_model_info(self, model: str = 'local-model') -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model (str): The name of the model to get information about.

        Returns:
            Dict[str, Any]: Information about the model.

        Note:
            LM Studio doesn't provide an API to get model info, so this returns placeholder data.
        """
        logger.info(f"Getting model info for: {model}")
        # LM Studio doesn't provide an API to get model info, so we return placeholder data
        return {
            "id": model,
            "name": "Local LM Studio Model",
            "description": "A locally hosted language model through LM Studio"
        }