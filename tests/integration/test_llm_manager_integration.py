# tests/integration/test_llm_manager_integration.py

import pytest
from unittest.mock import patch, Mock
from llm_manager import LLMManager
from llm_providers import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider, LMStudioProvider, AzureOpenAIProvider, AWSBedrockProvider
import openai
import anthropic
import google.generativeai as genai
import requests
import boto3

@pytest.fixture
def mock_providers():
    with patch('llm_providers.openai_provider.OpenAIProvider') as mock_openai, \
         patch('llm_providers.anthropic_provider.AnthropicProvider') as mock_anthropic, \
         patch('llm_providers.gemini_provider.GeminiProvider') as mock_gemini, \
         patch('llm_providers.ollama_provider.OllamaProvider') as mock_ollama, \
         patch('llm_providers.lmstudio_provider.LMStudioProvider') as mock_lmstudio, \
         patch('llm_providers.azure_openai_provider.AzureOpenAIProvider') as mock_azure, \
         patch('llm_providers.aws_bedrock_provider.AWSBedrockProvider') as mock_aws:
        yield {
            'openai': mock_openai,
            'anthropic': mock_anthropic,
            'gemini': mock_gemini,
            'ollama': mock_ollama,
            'lmstudio': mock_lmstudio,
            'azure_openai': mock_azure,
            'aws_bedrock': mock_aws
        }

@pytest.fixture
def llm_manager(mock_providers):
    return LLMManager(providers=['openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'])

def test_generate_integration(llm_manager, mock_providers):
    for provider in mock_providers.values():
        provider.return_value.generate.return_value = f"Generated text from {provider.__name__}"

    result = llm_manager.generate("Test prompt")
    
    assert len(result) == 7  # One result for each provider
    for provider_name, provider_result in result.items():
        assert provider_result['status'] == 'success'
        assert f"Generated text from" in provider_result['result']
        mock_providers[provider_name].return_value.generate.assert_called_once_with("Test prompt", max_tokens=150, temperature=0.7)

def test_generate_with_error(llm_manager, mock_providers):
    mock_providers['openai'].return_value.generate.side_effect = openai.OpenAIError("API Error")
    mock_providers['anthropic'].return_value.generate.return_value = "Generated text from Anthropic"

    result = llm_manager.generate("Test prompt")

    assert result['openai']['status'] == 'error'
    assert "API Error" in result['openai']['error']
    assert result['anthropic']['status'] == 'success'
    assert result['anthropic']['result'] == "Generated text from Anthropic"

def test_generate_stream_integration(llm_manager, mock_providers):
    mock_providers['gemini'].return_value.generate_stream.return_value = iter(["Chunk 1", "Chunk 2", "Chunk 3"])

    result = list(llm_manager.generate_stream("Test prompt", provider='gemini'))

    assert len(result) == 3
    assert all(chunk['status'] == 'success' for chunk in result)
    assert [chunk['chunk'] for chunk in result] == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_providers['gemini'].return_value.generate_stream.assert_called_once_with("Test prompt", max_tokens=150, temperature=0.7)

def test_embed_integration(llm_manager, mock_providers):
    mock_providers['openai'].return_value.embed.return_value = [0.1, 0.2, 0.3]
    mock_providers['aws_bedrock'].return_value.embed.return_value = [0.4, 0.5, 0.6]

    result = llm_manager.embed("Test text", provider=['openai', 'aws_bedrock'])

    assert len(result) == 2
    assert result['openai']['status'] == 'success'
    assert result['openai']['result'] == [0.1, 0.2, 0.3]
    assert result['aws_bedrock']['status'] == 'success'
    assert result['aws_bedrock']['result'] == [0.4, 0.5, 0.6]

def test_chat_integration(llm_manager, mock_providers):
    mock_providers['anthropic'].return_value.chat.return_value = {"role": "assistant", "content": "Chat response from Anthropic"}
    mock_providers['azure_openai'].return_value.chat.return_value = {"role": "assistant", "content": "Chat response from Azure OpenAI"}

    messages = [{"role": "user", "content": "Hello"}]
    result = llm_manager.chat(messages, provider=['anthropic', 'azure_openai'])

    assert len(result) == 2
    assert result['anthropic']['status'] == 'success'
    assert result['anthropic']['result']['content'] == "Chat response from Anthropic"
    assert result['azure_openai']['status'] == 'success'
    assert result['azure_openai']['result']['content'] == "Chat response from Azure OpenAI"

def test_generate_parallel_integration(llm_manager, mock_providers):
    for provider in mock_providers.values():
        provider.return_value.generate.side_effect = [
            f"Generated text 1 from {provider.__name__}",
            f"Generated text 2 from {provider.__name__}"
        ]

    prompts = ["Prompt 1", "Prompt 2"]
    result = llm_manager.generate_parallel(prompts)

    assert len(result) == 2
    for prompt_result in result.values():
        assert len(prompt_result) == 7  # One result for each provider
        for provider_name, provider_result in prompt_result.items():
            assert provider_result['status'] == 'success'
            assert f"Generated text" in provider_result['result']
            assert f"from {provider_name.capitalize()}" in provider_result['result']

def test_get_provider_info_integration(llm_manager, mock_providers):
    for provider in mock_providers.values():
        provider.return_value.list_available_models.return_value = [{"name": f"Model1-{provider.__name__}", "version": "1.0"}]

    info = llm_manager.get_provider_info()

    assert len(info) == 7  # One info entry for each provider
    for provider_name, provider_info in info.items():
        assert 'name' in provider_info
        assert 'type' in provider_info
        assert 'methods' in provider_info
        assert 'available_models' in provider_info
        assert len(provider_info['available_models']) == 1
        assert provider_info['available_models'][0]['name'] == f"Model1-{provider_name.capitalize()}Provider"

def test_generate_with_custom_parameters(llm_manager, mock_providers):
    mock_providers['openai'].return_value.generate.return_value = "Generated with custom parameters"

    result = llm_manager.generate(
        "Test prompt",
        provider='openai',
        model="gpt-4",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95
    )

    assert result['openai']['status'] == 'success'
    assert result['openai']['result'] == "Generated with custom parameters"
    mock_providers['openai'].return_value.generate.assert_called_once_with(
        "Test prompt",
        model="gpt-4",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95
    )

def test_error_handling_integration(llm_manager, mock_providers):
    mock_providers['openai'].return_value.generate.side_effect = openai.OpenAIError("OpenAI API Error")
    mock_providers['anthropic'].return_value.generate.side_effect = anthropic.APIError("Anthropic API Error")
    mock_providers['gemini'].return_value.generate.side_effect = Exception("Gemini Error")
    mock_providers['ollama'].return_value.generate.side_effect = requests.RequestException("Ollama Request Error")
    mock_providers['lmstudio'].return_value.generate.side_effect = openai.OpenAIError("LM Studio Error")
    mock_providers['azure_openai'].return_value.generate.side_effect = openai.OpenAIError("Azure OpenAI Error")
    mock_providers['aws_bedrock'].return_value.generate.side_effect = boto3.exceptions.Boto3Error("AWS Bedrock Error")

    result = llm_manager.generate("Test prompt")

    for provider_name, provider_result in result.items():
        assert provider_result['status'] == 'error'
        assert f"{provider_name.upper()} Error" in provider_result['error'] or "API Error" in provider_result['error']

def test_generate_stream_error_handling(llm_manager, mock_providers):
    mock_providers['gemini'].return_value.generate_stream.side_effect = Exception("Stream Error")

    result = list(llm_manager.generate_stream("Test prompt", provider='gemini'))

    assert len(result) == 1
    assert result[0]['status'] == 'error'
    assert "Stream Error" in result[0]['error']

@patch('llm_manager.ThreadPoolExecutor')
def test_generate_parallel_with_timeout(mock_executor, llm_manager, mock_providers):
    def slow_generate(*args, **kwargs):
        import time
        time.sleep(2)
        return "Slow response"

    mock_providers['openai'].return_value.generate.side_effect = slow_generate
    mock_executor.return_value.__enter__.return_value.submit.side_effect = [
        Mock(result=lambda: {"openai": {"status": "success", "result": "Fast response"}}),
        Mock(side_effect=TimeoutError("Operation timed out"))
    ]

    prompts = ["Fast prompt", "Slow prompt"]
    result = llm_manager.generate_parallel(prompts, timeout=1)

    assert len(result) == 2
    assert result[0]['openai']['status'] == 'success'
    assert result[0]['openai']['result'] == "Fast response"
    assert result[1]['openai']['status'] == 'error'
    assert "Operation timed out" in result[1]['openai']['error']

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])