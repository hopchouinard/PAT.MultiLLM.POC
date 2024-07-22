# llm_providers/azure_openai_provider.py

import openai
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the Azure OpenAI provider with API details from configuration.
        """
        config = get_config()
        self.api_key = config.get('azure_openai_api_key')
        self.api_base = config.get('azure_openai_api_base')
        self.api_version = config.get('azure_openai_api_version', '2023-05-15')
        
        if not self.api_key or not self.api_base:
            logger.error("Azure OpenAI API key or base URL not found in configuration")
            raise ValueError("Azure OpenAI API key or base URL not found in configuration")
        
        openai.api_type = "azure"
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_key = self.api_key
        
        logger.info("AzureOpenAIProvider initialized successfully")

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
        deployment_name = kwargs.get('deployment_name', 'text-davinci-003')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with deployment: {deployment_name}")
        try:
            response = openai.Completion.create(
                engine=deployment_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except openai.error.OpenAIError as e:
            logger.error(f"Azure OpenAI API error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Azure OpenAI API call: {str(e)}", exc_info=True)
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
        deployment_name = kwargs.get('deployment_name', 'text-davinci-003')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with deployment: {deployment_name}")
        try:
            stream = openai.Completion.create(
                engine=deployment_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].text:
                    yield chunk.choices[0].text
        except openai.error.OpenAIError as e:
            logger.error(f"Azure OpenAI API streaming error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Azure OpenAI API streaming call: {str(e)}", exc_info=True)
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
        deployment_name = kwargs.get('deployment_name', 'text-embedding-ada-002')
        
        logger.info(f"Generating embedding with deployment: {deployment_name}")
        try:
            response = openai.Embedding.create(
                input=[text],
                engine=deployment_name
            )
            return response['data'][0]['embedding']
        except openai.error.OpenAIError as e:
            logger.error(f"Azure OpenAI API embedding error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Azure OpenAI embedding: {str(e)}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the Azure OpenAI model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'assistant') and 'content' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        deployment_name = kwargs.get('deployment_name', 'gpt-35-turbo')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with deployment: {deployment_name}")
        try:
            response = openai.ChatCompletion.create(
                engine=deployment_name,
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
            logger.error(f"Azure OpenAI API chat error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Azure OpenAI API chat call: {str(e)}", exc_info=True)
            raise

    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List available deployments from Azure OpenAI.

        Returns:
            List[Dict[str, Any]]: A list of available deployments and their details.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info("Fetching available deployments from Azure OpenAI")
        try:
            # Note: As of my knowledge cutoff, there's no direct API to list deployments.
            # This would typically involve calling an Azure-specific API or management interface.
            # For now, we'll return a placeholder message.
            return [{"message": "Deployment listing not directly supported via OpenAI API. Please check your Azure portal or use Azure SDK for deployment management."}]
        except Exception as e:
            logger.error(f"Error fetching Azure OpenAI deployments: {str(e)}", exc_info=True)
            raise

    def get_deployment_info(self, deployment_name: str) -> Dict[str, Any]:
        """
        Get information about a specific deployment.

        Args:
            deployment_name (str): The name of the deployment to retrieve information for.

        Returns:
            Dict[str, Any]: Information about the deployment.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info(f"Getting deployment info for: {deployment_name}")
        try:
            # Note: As of my knowledge cutoff, there's no direct API to get deployment info.
            # This would typically involve calling an Azure-specific API or management interface.
            # For now, we'll return a placeholder message.
            return {"message": f"Deployment info for {deployment_name} not directly accessible via OpenAI API. Please check your Azure portal or use Azure SDK for deployment management."}
        except Exception as e:
            logger.error(f"Error retrieving Azure OpenAI deployment info: {str(e)}", exc_info=True)
            raise