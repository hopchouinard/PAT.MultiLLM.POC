# tests/unit/test_providers/test_azure_openai_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.azure_openai_provider import AzureOpenAIProvider
import openai

@pytest.fixture
def mock_openai():
    with patch('llm_providers.azure_openai_provider.openai') as mock:
        yield mock

@pytest.fixture
def azure_openai_provider(mock_openai, mock_config):
    return AzureOpenAIProvider()

def test_azure_openai_provider_initialization(azure_openai_provider, mock_config, mock_openai):
    assert isinstance(azure_openai_provider, AzureOpenAIProvider)
    assert azure_openai_provider.api_key == 'test_azure_openai_key'
    assert azure_openai_provider.api_base == 'https://test.openai.azure.com/'
    assert azure_openai_provider.api_version == '2023-05-15'
    assert mock_openai.api_type == "azure"
    assert mock_openai.api_base == 'https://test.openai.azure.com/'
    assert mock_openai.api_version == '2023-05-15'
    assert mock_openai.api_key == 'test_azure_openai_key'

def test_generate(azure_openai_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].text = "Generated text"
    mock_openai.Completion.create.return_value = mock_completion

    result = azure_openai_provider.generate("Test prompt", deployment_name="text-davinci-003", max_tokens=10, temperature=0.7)
    assert result == "Generated text"
    mock_openai.Completion.create.assert_called_once_with(
        engine="text-davinci-003",
        prompt="Test prompt",
        max_tokens=10,
        temperature=0.7
    )

def test_generate_chat_model(azure_openai_provider, mock_openai):
    mock_chat_completion = Mock()
    mock_chat_completion.choices[0].message.content = "Chat generated text"
    mock_openai.ChatCompletion.create.return_value = mock_chat_completion

    result = azure_openai_provider.generate("Test prompt", deployment_name="gpt-35-turbo", max_tokens=10, temperature=0.7)
    assert result == "Chat generated text"
    mock_openai.ChatCompletion.create.assert_called_once_with(
        engine="gpt-35-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        temperature=0.7
    )

def test_generate_api_error(azure_openai_provider, mock_openai):
    mock_openai.Completion.create.side_effect = openai.OpenAIError("API Error")
    with pytest.raises(openai.OpenAIError, match="API Error"):
        azure_openai_provider.generate("Test prompt", deployment_name="text-davinci-003")

def test_generate_stream(azure_openai_provider, mock_openai):
    mock_stream = Mock()
    mock_stream.__iter__.return_value = [
        Mock(choices=[Mock(text="Chunk 1")]),
        Mock(choices=[Mock(text="Chunk 2")]),
        Mock(choices=[Mock(text="Chunk 3")])
    ]
    mock_openai.Completion.create.return_value = mock_stream

    result = list(azure_openai_provider.generate_stream("Test prompt", deployment_name="text-davinci-003", max_tokens=30, temperature=0.8))
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_openai.Completion.create.assert_called_once_with(
        engine="text-davinci-003",
        prompt="Test prompt",
        max_tokens=30,
        temperature=0.8,
        stream=True
    )

def test_generate_stream_chat_model(azure_openai_provider, mock_openai):
    mock_stream = Mock()
    mock_stream.__iter__.return_value = [
        Mock(choices=[Mock(delta=Mock(content="Chunk A"))]),
        Mock(choices=[Mock(delta=Mock(content="Chunk B"))]),
        Mock(choices=[Mock(delta=Mock(content="Chunk C"))])
    ]
    mock_openai.ChatCompletion.create.return_value = mock_stream

    result = list(azure_openai_provider.generate_stream("Test prompt", deployment_name="gpt-35-turbo", max_tokens=30, temperature=0.8))
    assert result == ["Chunk A", "Chunk B", "Chunk C"]
    mock_openai.ChatCompletion.create.assert_called_once_with(
        engine="gpt-35-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=30,
        temperature=0.8,
        stream=True
    )

def test_generate_stream_error(azure_openai_provider, mock_openai):
    mock_openai.Completion.create.side_effect = openai.OpenAIError("Stream Error")
    with pytest.raises(openai.OpenAIError, match="Stream Error"):
        list(azure_openai_provider.generate_stream("Test prompt", deployment_name="text-davinci-003"))

def test_embed(azure_openai_provider, mock_openai):
    mock_embedding = Mock()
    mock_embedding.data[0].embedding = [0.1, 0.2, 0.3]
    mock_openai.Embedding.create.return_value = mock_embedding

    result = azure_openai_provider.embed("Test text", deployment_name="text-embedding-ada-002")
    assert result == [0.1, 0.2, 0.3]
    mock_openai.Embedding.create.assert_called_once_with(
        input=["Test text"],
        engine="text-embedding-ada-002"
    )

def test_embed_error(azure_openai_provider, mock_openai):
    mock_openai.Embedding.create.side_effect = openai.OpenAIError("Embedding Error")
    with pytest.raises(openai.OpenAIError, match="Embedding Error"):
        azure_openai_provider.embed("Test text", deployment_name="text-embedding-ada-002")

def test_chat(azure_openai_provider, mock_openai):
    mock_chat_completion = Mock()
    mock_chat_completion.choices[0].message.content = "Chat response"
    mock_chat_completion.model = "gpt-35-turbo"
    mock_chat_completion.usage.prompt_tokens = 10
    mock_chat_completion.usage.completion_tokens = 20
    mock_chat_completion.usage.total_tokens = 30
    mock_openai.ChatCompletion.create.return_value = mock_chat_completion

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = azure_openai_provider.chat(messages, deployment_name="gpt-35-turbo", max_tokens=50, temperature=0.7)

    assert result == {
        "role": "assistant",
        "content": "Chat response",
        "model": "gpt-35-turbo",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
    mock_openai.ChatCompletion.create.assert_called_once_with(
        engine="gpt-35-turbo",
        messages=messages,
        max_tokens=50,
        temperature=0.7
    )

def test_chat_api_error(azure_openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.side_effect = openai.OpenAIError("Chat API Error")
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(openai.OpenAIError, match="Chat API Error"):
        azure_openai_provider.chat(messages, deployment_name="gpt-35-turbo")

def test_list_deployments(azure_openai_provider):
    deployments = azure_openai_provider.list_deployments()
    assert isinstance(deployments, list)
    assert len(deployments) == 1
    assert "message" in deployments[0]
    assert "Deployment listing not directly supported" in deployments[0]["message"]

def test_get_deployment_info(azure_openai_provider):
    info = azure_openai_provider.get_deployment_info("gpt-35-turbo")
    assert isinstance(info, dict)
    assert "message" in info
    assert "Deployment info for gpt-35-turbo not directly accessible" in info["message"]

def test_generate_with_custom_parameters(azure_openai_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].text = "Generated with custom parameters"
    mock_openai.Completion.create.return_value = mock_completion

    result = azure_openai_provider.generate(
        "Test prompt",
        deployment_name="text-davinci-003",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )
    assert result == "Generated with custom parameters"
    mock_openai.Completion.create.assert_called_once_with(
        engine="text-davinci-003",
        prompt="Test prompt",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )

def test_chat_with_system_message(azure_openai_provider, mock_openai):
    mock_chat_completion = Mock()
    mock_chat_completion.choices[0].message.content = "Chat response with system message"
    mock_chat_completion.model = "gpt-35-turbo"
    mock_openai.ChatCompletion.create.return_value = mock_chat_completion

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    result = azure_openai_provider.chat(messages, deployment_name="gpt-35-turbo")
    assert result["content"] == "Chat response with system message"
    mock_openai.ChatCompletion.create.assert_called_once()
    call_args = mock_openai.ChatCompletion.create.call_args[1]
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "You are a helpful assistant."

def test_generate_with_stop_sequences(azure_openai_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].text = "Generated text with stop"
    mock_openai.Completion.create.return_value = mock_completion

    result = azure_openai_provider.generate(
        "Test prompt",
        deployment_name="text-davinci-003",
        stop=["END", "STOP"]
    )
    assert result == "Generated text with stop"
    mock_openai.Completion.create.assert_called_once_with(
        engine="text-davinci-003",
        prompt="Test prompt",
        max_tokens=150,
        temperature=0.7,
        stop=["END", "STOP"]
    )

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])