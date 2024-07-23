# tests/integration/test_provider_integration.py

import pytest
import os
import openai
import anthropic
import google.generativeai as genai
import ollama
import requests
import boto3
from botocore.exceptions import ClientError
from llm_providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    LMStudioProvider,
    AzureOpenAIProvider,
    AWSBedrockProvider
)

# Setup test configuration
@pytest.fixture(scope="module")
def test_config():
    return {
        "openai_api_key": os.getenv("TEST_OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("TEST_ANTHROPIC_API_KEY"),
        "gemini_api_key": os.getenv("TEST_GEMINI_API_KEY"),
        "azure_openai_api_key": os.getenv("TEST_AZURE_OPENAI_API_KEY"),
        "azure_openai_api_base": os.getenv("TEST_AZURE_OPENAI_API_BASE"),
        "aws_access_key_id": os.getenv("TEST_AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("TEST_AWS_SECRET_ACCESS_KEY"),
        "aws_region_name": os.getenv("TEST_AWS_REGION_NAME"),
        "ollama_host": os.getenv("TEST_OLLAMA_HOST", "http://localhost:11434"),
        "lmstudio_api_base": os.getenv("TEST_LMSTUDIO_API_BASE", "http://localhost:1234/v1"),
    }

# Provider fixtures
@pytest.fixture
def openai_provider(test_config):
    return OpenAIProvider()

@pytest.fixture
def anthropic_provider(test_config):
    return AnthropicProvider()

@pytest.fixture
def gemini_provider(test_config):
    return GeminiProvider()

@pytest.fixture
def ollama_provider(test_config):
    return OllamaProvider()

@pytest.fixture
def lmstudio_provider(test_config):
    return LMStudioProvider()

@pytest.fixture
def azure_openai_provider(test_config):
    return AzureOpenAIProvider()

@pytest.fixture
def aws_bedrock_provider(test_config):
    return AWSBedrockProvider()

# Test OpenAI Provider
@pytest.mark.skipif(not os.getenv("TEST_OPENAI_API_KEY"), reason="OpenAI API key not provided")
def test_openai_generate(openai_provider):
    result = openai_provider.generate("What is the capital of France?", model="gpt-4o-mini")
    assert isinstance(result, str)
    assert "Paris" in result.lower()

@pytest.mark.skipif(not os.getenv("TEST_OPENAI_API_KEY"), reason="OpenAI API key not provided")
def test_openai_embed(openai_provider):
    result = openai_provider.embed("This is a test sentence.")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

@pytest.mark.skipif(not os.getenv("TEST_OPENAI_API_KEY"), reason="OpenAI API key not provided")
def test_openai_chat(openai_provider):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    result = openai_provider.chat(messages, model="gpt-4o-mini")
    assert isinstance(result, dict)
    assert "content" in result
    assert isinstance(result["content"], str)

# Test Anthropic Provider
@pytest.mark.skipif(not os.getenv("TEST_ANTHROPIC_API_KEY"), reason="Anthropic API key not provided")
def test_anthropic_generate(anthropic_provider):
    result = anthropic_provider.generate("Explain quantum computing briefly.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("TEST_ANTHROPIC_API_KEY"), reason="Anthropic API key not provided")
def test_anthropic_chat(anthropic_provider):
    messages = [{"role": "user", "content": "What are the main types of machine learning?"}]
    result = anthropic_provider.chat(messages)
    assert isinstance(result, dict)
    assert "content" in result
    assert isinstance(result["content"], str)

# Test Gemini Provider
@pytest.mark.skipif(not os.getenv("TEST_GEMINI_API_KEY"), reason="Gemini API key not provided")
def test_gemini_generate(gemini_provider):
    result = gemini_provider.generate("What are the benefits of renewable energy?")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("TEST_GEMINI_API_KEY"), reason="Gemini API key not provided")
def test_gemini_embed(gemini_provider):
    result = gemini_provider.embed("This is a test sentence for Gemini embedding.")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

# Test Ollama Provider
@pytest.mark.skipif(not os.getenv("TEST_OLLAMA_HOST"), reason="Ollama host not provided")
def test_ollama_generate(ollama_provider):
    result = ollama_provider.generate("What is the significance of the number 42?")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("TEST_OLLAMA_HOST"), reason="Ollama host not provided")
def test_ollama_embed(ollama_provider):
    result = ollama_provider.embed("This is a test sentence for Ollama embedding.")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

# Test LM Studio Provider
@pytest.mark.skipif(not os.getenv("TEST_LMSTUDIO_API_BASE"), reason="LM Studio API base not provided")
def test_lmstudio_generate(lmstudio_provider):
    result = lmstudio_provider.generate("Explain the concept of artificial intelligence.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("TEST_LMSTUDIO_API_BASE"), reason="LM Studio API base not provided")
def test_lmstudio_chat(lmstudio_provider):
    messages = [{"role": "user", "content": "What are the three laws of robotics?"}]
    result = lmstudio_provider.chat(messages)
    assert isinstance(result, dict)
    assert "content" in result
    assert isinstance(result["content"], str)

# Test Azure OpenAI Provider
@pytest.mark.skipif(not os.getenv("TEST_AZURE_OPENAI_API_KEY"), reason="Azure OpenAI API key not provided")
def test_azure_openai_generate(azure_openai_provider):
    result = azure_openai_provider.generate("What is the importance of sustainable development?", deployment_name="your-deployment-name")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("TEST_AZURE_OPENAI_API_KEY"), reason="Azure OpenAI API key not provided")
def test_azure_openai_embed(azure_openai_provider):
    result = azure_openai_provider.embed("This is a test sentence for Azure OpenAI embedding.", deployment_name="your-embedding-deployment")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

# Test AWS Bedrock Provider
@pytest.mark.skipif(not os.getenv("TEST_AWS_ACCESS_KEY_ID"), reason="AWS credentials not provided")
def test_aws_bedrock_generate(aws_bedrock_provider):
    result = aws_bedrock_provider.generate("Explain the concept of cloud computing.", model_id="anthropic.claude-v2")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("TEST_AWS_ACCESS_KEY_ID"), reason="AWS credentials not provided")
def test_aws_bedrock_embed(aws_bedrock_provider):
    result = aws_bedrock_provider.embed("This is a test sentence for AWS Bedrock embedding.", model_id="amazon.titan-embed-text-v1")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

# Error handling tests
def test_openai_error_handling(openai_provider):
    with pytest.raises(openai.OpenAIError):
        openai_provider.generate("Test prompt", model="non-existent-model")

def test_anthropic_error_handling(anthropic_provider):
    with pytest.raises(anthropic.APIError):
        anthropic_provider.generate("Test prompt", model="non-existent-model")

def test_gemini_error_handling(gemini_provider):
    with pytest.raises(Exception):  # Adjust this to the specific exception type Gemini uses
        gemini_provider.generate("Test prompt", model="non-existent-model")

def test_ollama_error_handling(ollama_provider):
    with pytest.raises(requests.RequestException):
        ollama_provider.generate("Test prompt", model="non-existent-model")

def test_lmstudio_error_handling(lmstudio_provider):
    with pytest.raises(openai.OpenAIError):
        lmstudio_provider.generate("Test prompt", model="non-existent-model")

def test_azure_openai_error_handling(azure_openai_provider):
    with pytest.raises(openai.OpenAIError):
        azure_openai_provider.generate("Test prompt", deployment_name="non-existent-deployment")

def test_aws_bedrock_error_handling(aws_bedrock_provider):
    with pytest.raises(ClientError):
        aws_bedrock_provider.generate("Test prompt", model_id="non-existent-model")

# Additional tests for specific provider features
@pytest.mark.skipif(not os.getenv("TEST_OPENAI_API_KEY"), reason="OpenAI API key not provided")
def test_openai_streaming(openai_provider):
    chunks = list(openai_provider.generate_stream("Tell me a short story.", model="gpt-4o-mini"))
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    full_response = ''.join(chunks)
    assert len(full_response) > 0

@pytest.mark.skipif(not os.getenv("TEST_ANTHROPIC_API_KEY"), reason="Anthropic API key not provided")
def test_anthropic_with_system_prompt(anthropic_provider):
    result = anthropic_provider.generate("What's your primary function?", system_prompt="You are a helpful AI assistant created by Anthropic.")
    assert isinstance(result, str)
    assert "Anthropic" in result or "assistant" in result.lower()

@pytest.mark.skipif(not os.getenv("TEST_GEMINI_API_KEY"), reason="Gemini API key not provided")
def test_gemini_safety_settings(gemini_provider):
    gemini_provider.set_safety_settings(harassment=genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE)
    result = gemini_provider.generate("Tell me a controversial opinion.")
    assert isinstance(result, str)
    # Check that the response is moderated (this is subjective and may need adjustment)
    assert len(result) > 0 and "I'm sorry" in result or "I don't feel comfortable" in result

@pytest.mark.skipif(not os.getenv("TEST_OLLAMA_HOST"), reason="Ollama host not provided")
def test_ollama_model_management(ollama_provider):
    models_before = ollama_provider.list_available_models()
    ollama_provider.pull_model("llama2")
    models_after = ollama_provider.list_available_models()
    assert len(models_after) > len(models_before)
    assert any(model['name'] == 'llama2' for model in models_after)

@pytest.mark.skipif(not os.getenv("TEST_AWS_ACCESS_KEY_ID"), reason="AWS credentials not provided")
def test_aws_bedrock_model_info(aws_bedrock_provider):
    model_info = aws_bedrock_provider.get_model_info("anthropic.claude-v2")
    assert isinstance(model_info, dict)
    assert "modelId" in model_info
    assert model_info["modelId"] == "anthropic.claude-v2"

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])