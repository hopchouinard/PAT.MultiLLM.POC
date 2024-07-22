# tests/conftest.py

import pytest
from unittest.mock import Mock, patch
from llm_manager import LLMManager
from llm_providers import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider, LMStudioProvider, AzureOpenAIProvider, AWSBedrockProvider
from utils.config import Config

@pytest.fixture(scope="session")
def mock_config():
    config = Config()
    config.set('openai_api_key', 'test_openai_key')
    config.set('anthropic_api_key', 'test_anthropic_key')
    config.set('gemini_api_key', 'test_gemini_key')
    config.set('azure_openai_api_key', 'test_azure_openai_key')
    config.set('azure_openai_api_base', 'https://test.openai.azure.com/')
    config.set('aws_access_key_id', 'test_aws_access_key_id')
    config.set('aws_secret_access_key', 'test_aws_secret_access_key')
    config.set('aws_region_name', 'us-west-2')
    config.set('ollama_host', 'http://localhost:11434')
    config.set('lmstudio_api_base', 'http://localhost:1234/v1')
    return config

@pytest.fixture
def mock_get_config(mock_config):
    with patch('utils.config.get_config', return_value=mock_config):
        yield mock_config

@pytest.fixture
def mock_openai_provider():
    with patch('llm_providers.openai_provider.openai') as mock_openai:
        provider = OpenAIProvider()
        provider.generate = Mock(return_value="OpenAI generated text")
        provider.generate_stream = Mock(return_value=iter(["OpenAI ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[0.1, 0.2, 0.3])
        provider.chat = Mock(return_value={"role": "assistant", "content": "OpenAI chat response"})
        yield provider

@pytest.fixture
def mock_anthropic_provider():
    with patch('llm_providers.anthropic_provider.anthropic') as mock_anthropic:
        provider = AnthropicProvider()
        provider.generate = Mock(return_value="Anthropic generated text")
        provider.generate_stream = Mock(return_value=iter(["Anthropic ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[0.4, 0.5, 0.6])
        provider.chat = Mock(return_value={"role": "assistant", "content": "Anthropic chat response"})
        yield provider

@pytest.fixture
def mock_gemini_provider():
    with patch('llm_providers.gemini_provider.google.generativeai') as mock_genai:
        provider = GeminiProvider()
        provider.generate = Mock(return_value="Gemini generated text")
        provider.generate_stream = Mock(return_value=iter(["Gemini ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[0.7, 0.8, 0.9])
        provider.chat = Mock(return_value={"role": "model", "content": "Gemini chat response"})
        yield provider

@pytest.fixture
def mock_ollama_provider():
    with patch('llm_providers.ollama_provider.ollama') as mock_ollama:
        provider = OllamaProvider()
        provider.generate = Mock(return_value="Ollama generated text")
        provider.generate_stream = Mock(return_value=iter(["Ollama ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[1.0, 1.1, 1.2])
        provider.chat = Mock(return_value={"role": "assistant", "content": "Ollama chat response"})
        yield provider

@pytest.fixture
def mock_lmstudio_provider():
    with patch('llm_providers.lmstudio_provider.openai') as mock_openai:
        provider = LMStudioProvider()
        provider.generate = Mock(return_value="LM Studio generated text")
        provider.generate_stream = Mock(return_value=iter(["LM Studio ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[1.3, 1.4, 1.5])
        provider.chat = Mock(return_value={"role": "assistant", "content": "LM Studio chat response"})
        yield provider

@pytest.fixture
def mock_azure_openai_provider():
    with patch('llm_providers.azure_openai_provider.openai') as mock_openai:
        provider = AzureOpenAIProvider()
        provider.generate = Mock(return_value="Azure OpenAI generated text")
        provider.generate_stream = Mock(return_value=iter(["Azure OpenAI ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[1.6, 1.7, 1.8])
        provider.chat = Mock(return_value={"role": "assistant", "content": "Azure OpenAI chat response"})
        yield provider

@pytest.fixture
def mock_aws_bedrock_provider():
    with patch('llm_providers.aws_bedrock_provider.boto3') as mock_boto3:
        provider = AWSBedrockProvider()
        provider.generate = Mock(return_value="AWS Bedrock generated text")
        provider.generate_stream = Mock(return_value=iter(["AWS Bedrock ", "streamed ", "text"]))
        provider.embed = Mock(return_value=[1.9, 2.0, 2.1])
        provider.chat = Mock(return_value={"role": "assistant", "content": "AWS Bedrock chat response"})
        yield provider

@pytest.fixture
def llm_manager(mock_openai_provider, mock_anthropic_provider, mock_gemini_provider, 
                mock_ollama_provider, mock_lmstudio_provider, mock_azure_openai_provider, 
                mock_aws_bedrock_provider):
    with patch('llm_manager.get_provider') as mock_get_provider:
        mock_get_provider.side_effect = [
            mock_openai_provider,
            mock_anthropic_provider,
            mock_gemini_provider,
            mock_ollama_provider,
            mock_lmstudio_provider,
            mock_azure_openai_provider,
            mock_aws_bedrock_provider
        ]
        return LLMManager(providers=['openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'])

@pytest.fixture
def sample_text():
    return "This is a sample text for testing purposes."

@pytest.fixture
def sample_chat_messages():
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm an AI language model, so I don't have feelings, but I'm functioning well. How can I assist you today?"},
        {"role": "user", "content": "Can you explain what artificial intelligence is?"}
    ]

# Add more fixtures as needed for specific test scenarios