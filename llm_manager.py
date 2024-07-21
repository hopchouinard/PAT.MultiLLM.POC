# llm_manager.py

from typing import List, Dict, Any, Generator
from llm_providers import get_provider, list_supported_providers, LLMProvider
from utils import logger, get_config
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMManager:
    """
    Manager class for handling multiple LLM providers.
    """

    def __init__(self, providers: List[str] = None):
        """
        Initialize the LLMManager with specified providers.

        Args:
            providers (List[str], optional): List of provider names to initialize.
                If None, all supported providers will be initialized.
        """
        self.config = get_config()
        self.providers: Dict[str, LLMProvider] = {}

        if providers is None:
            providers = list_supported_providers()

        for provider_name in providers:
            try:
                self.providers[provider_name] = get_provider(provider_name)
                logger.info(f"Initialized {provider_name} provider")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} provider: {str(e)}")

    def generate(self, prompt: str, provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text using the specified provider or all providers.

        Args:
            prompt (str): The input prompt for text generation.
            provider (str, optional): The name of the provider to use. If None, all providers are used.
            **kwargs: Additional arguments to pass to the provider's generate method.

        Returns:
            Dict[str, Any]: A dictionary containing the generated text and metadata for each provider.
        """
        if provider and provider not in self.providers:
            raise ValueError(f"Provider {provider} not initialized")

        providers_to_use = [provider] if provider else self.providers.keys()
        results = {}

        for prov in providers_to_use:
            try:
                response = self.providers[prov].generate(prompt, **kwargs)
                results[prov] = {"response": response, "status": "success"}
            except Exception as e:
                logger.error(f"Error in {prov} generate: {str(e)}")
                results[prov] = {"response": None, "status": "error", "error": str(e)}

        return results

    def generate_stream(self, prompt: str, provider: str = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text in a streaming fashion using the specified provider.

        Args:
            prompt (str): The input prompt for text generation.
            provider (str, optional): The name of the provider to use. If None, the first available provider is used.
            **kwargs: Additional arguments to pass to the provider's generate_stream method.

        Yields:
            Dict[str, Any]: A dictionary containing the generated text chunk and metadata.
        """
        if provider:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not initialized")
            provider_to_use = provider
        else:
            provider_to_use = next(iter(self.providers))

        try:
            for chunk in self.providers[provider_to_use].generate_stream(prompt, **kwargs):
                yield {"provider": provider_to_use, "chunk": chunk, "status": "success"}
        except Exception as e:
            logger.error(f"Error in {provider_to_use} generate_stream: {str(e)}")
            yield {"provider": provider_to_use, "chunk": None, "status": "error", "error": str(e)}

    def embed(self, text: str, provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings using the specified provider or all providers.

        Args:
            text (str): The input text to embed.
            provider (str, optional): The name of the provider to use. If None, all providers are used.
            **kwargs: Additional arguments to pass to the provider's embed method.

        Returns:
            Dict[str, Any]: A dictionary containing the embeddings and metadata for each provider.
        """
        if provider and provider not in self.providers:
            raise ValueError(f"Provider {provider} not initialized")

        providers_to_use = [provider] if provider else self.providers.keys()
        results = {}

        for prov in providers_to_use:
            try:
                embedding = self.providers[prov].embed(text, **kwargs)
                results[prov] = {"embedding": embedding, "status": "success"}
            except Exception as e:
                logger.error(f"Error in {prov} embed: {str(e)}")
                results[prov] = {"embedding": None, "status": "error", "error": str(e)}

        return results

    def chat(self, messages: List[Dict[str, str]], provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation using the specified provider or all providers.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' and 'content' keys.
            provider (str, optional): The name of the provider to use. If None, all providers are used.
            **kwargs: Additional arguments to pass to the provider's chat method.

        Returns:
            Dict[str, Any]: A dictionary containing the chat response and metadata for each provider.
        """
        if provider and provider not in self.providers:
            raise ValueError(f"Provider {provider} not initialized")

        providers_to_use = [provider] if provider else self.providers.keys()
        results = {}

        for prov in providers_to_use:
            try:
                response = self.providers[prov].chat(messages, **kwargs)
                results[prov] = {"response": response, "status": "success"}
            except Exception as e:
                logger.error(f"Error in {prov} chat: {str(e)}")
                results[prov] = {"response": None, "status": "error", "error": str(e)}

        return results

    def generate_parallel(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using all providers in parallel.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional arguments to pass to the provider's generate method.

        Returns:
            Dict[str, Any]: A dictionary containing the generated text and metadata for each provider.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            future_to_provider = {executor.submit(provider.generate, prompt, **kwargs): name 
                                  for name, provider in self.providers.items()}
            for future in as_completed(future_to_provider):
                provider_name = future_to_provider[future]
                try:
                    response = future.result()
                    results[provider_name] = {"response": response, "status": "success"}
                except Exception as e:
                    logger.error(f"Error in {provider_name} generate: {str(e)}")
                    results[provider_name] = {"response": None, "status": "error", "error": str(e)}
        return results

    def get_provider_info(self, provider: str = None) -> Dict[str, Any]:
        """
        Get information about the specified provider or all providers.

        Args:
            provider (str, optional): The name of the provider to get info for. If None, info for all providers is returned.

        Returns:
            Dict[str, Any]: A dictionary containing provider information.
        """
        if provider and provider not in self.providers:
            raise ValueError(f"Provider {provider} not initialized")

        providers_to_use = [provider] if provider else self.providers.keys()
        info = {}

        for prov in providers_to_use:
            try:
                provider_instance = self.providers[prov]
                info[prov] = {
                    "name": prov,
                    "type": type(provider_instance).__name__,
                    "methods": [method for method in dir(provider_instance) if callable(getattr(provider_instance, method)) and not method.startswith("_")]
                }
                if hasattr(provider_instance, 'list_available_models'):
                    info[prov]["available_models"] = provider_instance.list_available_models()
            except Exception as e:
                logger.error(f"Error getting info for {prov}: {str(e)}")
                info[prov] = {"error": str(e)}

        return info

# Example usage
if __name__ == "__main__":
    manager = LLMManager(["openai", "anthropic"])
    
    # Generate text
    results = manager.generate("Explain the concept of quantum computing in simple terms.")
    for provider, result in results.items():
        print(f"{provider}: {result['response']}")

    # Generate embeddings
    embed_results = manager.embed("Quantum computing is fascinating.")
    for provider, result in embed_results.items():
        if result['status'] == 'success':
            print(f"{provider} embedding dimension: {len(result['embedding'])}")

    # Chat
    chat_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What's a famous landmark there?"}
    ]
    chat_results = manager.chat(chat_messages)
    for provider, result in chat_results.items():
        if result['status'] == 'success':
            print(f"{provider} response: {result['response']['content']}")

    # Get provider info
    provider_info = manager.get_provider_info()
    print("Provider Info:", provider_info)