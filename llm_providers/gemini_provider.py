import google.generativeai as genai
from google.generativeai.types import SafetySettingDict, HarmCategory, HarmBlockThreshold
from google.generativeai.generative_models import GenerativeModel
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the Gemini provider with API key from configuration.
        """
        config = get_config()
        self.api_key = config.get('gemini_api_key')
        if not self.api_key:
            logger.error("Gemini API key not found in configuration")
            raise ValueError("Gemini API key not found in configuration")
        genai.configure(api_key=self.api_key)
        logger.info("GeminiProvider initialized successfully")

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
        model_name = kwargs.get('model', 'gemini-1.5-flash')
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with model: {model_name}")
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}", exc_info=True)
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
        model_name = kwargs.get('model', 'gemini-1.5-flash')
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with model: {model_name}")
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                stream=True
            )
            for chunk in response:
                yield chunk.text
        except Exception as e:
            logger.error(f"Gemini API streaming error: {str(e)}", exc_info=True)
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
        model_name = kwargs.get('model', 'embedding-001')
        logger.info(f"Generating embedding with model: {model_name}")
        try:
            model = genai.get_model(model_name)
            result = model.embed_content(text)
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini embedding error: {str(e)}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the Gemini model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'model') and 'parts' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        model_name = kwargs.get('model', 'gemini-1.5-flash')
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with model: {model_name}")
        try:
            model = GenerativeModel(model_name)
            chat = model.start_chat(history=[])
            
            for message in messages:
                role = message['role']
                content = message['content']
                if role == 'user':
                    chat.send_message(content)
                else:
                    # Add model messages to chat history
                    chat.history.append({"role": "model", "parts": [content]})

            response = chat.send_message(
                messages[-1]['content'],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            
            return {
                "role": "model",
                "content": response.text,
                "model": model_name,
            }
        except Exception as e:
            logger.error(f"Gemini API chat error: {str(e)}", exc_info=True)
            raise

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models from Gemini.

        Returns:
            List[Dict[str, Any]]: A list of available models and their details.
        """
        logger.info("Fetching available models from Gemini")
        try:
            models = genai.list_models()
            return [{"name": model.name, "description": model.description} for model in models]
        except Exception as e:
            logger.error(f"Error fetching Gemini models: {str(e)}", exc_info=True)
            raise

    def set_safety_settings(self, **kwargs) -> None:
        """
        Set safety settings for content generation.

        Args:
            **kwargs: Safety settings for different harm categories.

        Example:
            set_safety_settings(
                harassment=HarmBlockThreshold.MEDIUM_AND_ABOVE,
                hate_speech=HarmBlockThreshold.HIGH_AND_ABOVE,
                sexually_explicit=HarmBlockThreshold.MEDIUM_AND_ABOVE,
                dangerous_content=HarmBlockThreshold.MEDIUM_AND_ABOVE
            )
        """
        safety_settings: List[SafetySettingDict] = []
        for category, threshold in kwargs.items():
            if category in HarmCategory.__members__:
                safety_settings.append({
                    "category": HarmCategory[category],
                    "threshold": threshold
                })
            else:
                logger.warning(f"Unknown harm category: {category}")

        genai.configure(safety_settings=safety_settings)
        logger.info("Safety settings updated")