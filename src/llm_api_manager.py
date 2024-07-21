import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai
import ollama

# Load environment variables
load_dotenv()

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'gpt-4o-mini')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

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
        except Exception as e:
            raise Exception(f"Error in OpenAI API call: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        model = kwargs.get('model', 'gpt-4o-mini')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

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
        except Exception as e:
            raise Exception(f"Error in OpenAI API streaming call: {str(e)}")

    def embed(self, text: str, **kwargs) -> List[float]:
        model = kwargs.get('model', 'text-embedding-ada-002')
        try:
            response = openai.Embedding.create(input=[text], model=model)
            return response['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Error in OpenAI embedding: {str(e)}")

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found in environment variables")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'claude-3-5-sonnet-20240620')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        try:
            response = self.client.completions.create(
                model=model,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                max_tokens_to_sample=max_tokens,
                temperature=temperature
            )
            return response.completion.strip()
        except Exception as e:
            raise Exception(f"Error in Anthropic API call: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        model = kwargs.get('model', 'claude-3-5-sonnet-20240620')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        try:
            stream = self.client.completions.create(
                model=model,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.completion:
                    yield chunk.completion
        except Exception as e:
            raise Exception(f"Error in Anthropic API streaming call: {str(e)}")

    def embed(self, text: str, **kwargs) -> List[float]:
        # As of now, Anthropic doesn't provide a public embedding API
        # This is a placeholder for future implementation
        raise NotImplementedError("Embedding not currently supported for Anthropic")

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables")
        genai.configure(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model_name = kwargs.get('model', 'gemini-1.5-flash')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        try:
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Error in Gemini API call: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        model_name = kwargs.get('model', 'gemini-1.5-flash')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        try:
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise Exception(f"Error in Gemini API streaming call: {str(e)}")

    def embed(self, text: str, **kwargs) -> List[float]:
        model_name = kwargs.get('model', 'models/embedding-001')
        try:
            model = genai.GenerativeModel(model_name=model_name)
            embedding = model.embed_content(text)
            return embedding.values
        except Exception as e:
            raise Exception(f"Error in Gemini embedding: {str(e)}")

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama.set_host(self.host)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'llama3')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

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
        except Exception as e:
            raise Exception(f"Error in Ollama API call: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        model = kwargs.get('model', 'llama3')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

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
        except Exception as e:
            raise Exception(f"Error in Ollama API streaming call: {str(e)}")

    def embed(self, text: str, **kwargs) -> List[float]:
        model = kwargs.get('model', 'mxbai-embed-large')
        try:
            response = ollama.embeddings(model=model, prompt=text)
            return response['embedding']
        except Exception as e:
            raise Exception(f"Error in Ollama embedding: {str(e)}")

class LMStudioProvider(LLMProvider):
    def __init__(self):
        self.api_base = os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1')
        if not self.api_base:
            raise ValueError("LM Studio API base not found in environment variables")
        self.client = openai.OpenAI(base_url=self.api_base, api_key="not-needed")

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error in LM Studio API call: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        model = kwargs.get('model', 'lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

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
        except Exception as e:
            raise Exception(f"Error in LM Studio API streaming call: {str(e)}")

    def embed(self, text: str, **kwargs) -> List[float]:
        model = kwargs.get('model', 'nomic-embed-text-v1.5')
        try:
            response = self.client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error in LM Studio embedding: {str(e)}")

class LLMManager:
    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers

    def generate(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        results = []
        for provider in self.providers:
            try:
                response = provider.generate(prompt, **kwargs)
                results.append({
                    "provider": provider.__class__.__name__,
                    "response": response,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "provider": provider.__class__.__name__,
                    "response": None,
                    "status": "error",
                    "error": str(e)
                })
        return results

    def generate_stream(self, prompt: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        for provider in self.providers:
            try:
                for chunk in provider.generate_stream(prompt, **kwargs):
                    yield {
                        "provider": provider.__class__.__name__,
                        "chunk": chunk,
                        "status": "success"
                    }
            except Exception as e:
                yield {
                    "provider": provider.__class__.__name__,
                    "chunk": None,
                    "status": "error",
                    "error": str(e)
                }

    def embed(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        results = []
        for provider in self.providers:
            try:
                embedding = provider.embed(text, **kwargs)
                results.append({
                    "provider": provider.__class__.__name__,
                    "embedding": embedding,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "provider": provider.__class__.__name__,
                    "embedding": None,
                    "status": "error",
                    "error": str(e)
                })
        return results

def generate_llm_responses(prompt: str, providers: List[str], **kwargs) -> List[Dict[str, Any]]:
    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider
    }
    
    try:
        selected_providers = [provider_map[p.lower()]() for p in providers if p.lower() in provider_map]
    except ValueError as e:
        print(f"Error initializing provider: {str(e)}")
        return []

    manager = LLMManager(selected_providers)
    return manager.generate(prompt, **kwargs)

def generate_llm_responses_stream(prompt: str, providers: List[str], **kwargs) -> Generator[Dict[str, Any], None, None]:
    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider
    }
    
    try:
        selected_providers = [provider_map[p.lower()]() for p in providers if p.lower() in provider_map]
    except ValueError as e:
        print(f"Error initializing provider: {str(e)}")
        return

    manager = LLMManager(selected_providers)
    return manager.generate_stream(prompt, **kwargs)

def generate_embeddings(text: str, providers: List[str], **kwargs) -> List[Dict[str, Any]]:
    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider
    }
    
    try:
        selected_providers = [provider_map[p.lower()]() for p in providers if p.lower() in provider_map]
    except ValueError as e:
        print(f"Error initializing provider: {str(e)}")
        return []

    manager = LLMManager(selected_providers)
    return manager.embed(text, **kwargs)

# Example usage
if __name__ == "__main__":
    # Text generation example (non-streaming)
    prompt = "Explain the concept of recursion in programming."
    providers = ["openai", "anthropic", "gemini", "ollama", "lmstudio"]
    responses = generate_llm_responses(
        prompt, 
        providers, 
        model="local-model", 
        max_tokens=100, 
        temperature=0.7
    )
    
    for response in responses:
        print(f"Provider: {response['provider']}")
        print(f"Status: {response['status']}")
        if response['status'] == 'success':
            print(f"Response: {response['response']}\n")
        else:
            print(f"Error: {response['error']}\n")

    # Text generation example (streaming)
    print("\nStreaming responses:")
    stream = generate_llm_responses_stream(
        prompt, 
        providers, 
        model="local-model", 
        max_tokens=100, 
        temperature=0.7
    )
    
    for chunk in stream:
        print(f"Provider: {chunk['provider']}")
        print(f"Status: {chunk['status']}")
        if chunk['status'] == 'success':
            print(f"Chunk: {chunk['chunk']}", end='', flush=True)
        else:
            print(f"Error: {chunk['error']}")
    print()  # New line after streaming is complete

    # Embedding example
    text = "Explain the concept of embeddings in natural language processing."
    embedding_providers = ["openai", "gemini", "ollama", "lmstudio"]
    embeddings = generate_embeddings(text, embedding_providers)
    
    for result in embeddings:
        print(f"Provider: {result['provider']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Embedding shape: {len(result['embedding'])}")
            print(f"First 5 values: {result['embedding'][:5]}\n")
        else:
            print(f"Error: {result['error']}\n")