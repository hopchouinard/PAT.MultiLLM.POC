# tests/unit/test_providers/test_anthropic_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.anthropic_provider import AnthropicProvider
from anthropic import APIError, RateLimitError, AuthenticationError

@pytest.fixture
def mock_anthropic():
    with patch('llm_providers.anthropic_provider.anthropic') as mock:
        yield mock

@pytest.fixture
def anthropic_provider(mock_anthropic, mock_config):
    return AnthropicProvider()

def test_anthropic_provider_initialization(anthropic_provider, mock_config):
    assert isinstance(anthropic_provider, AnthropicProvider)
    assert anthropic_provider.api_key == 'test_anthropic_key'
    assert anthropic_provider.client is not None

def test_generate(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.return_value.completion = "Generated text"
    result = anthropic_provider.generate("Test prompt", max_tokens=10, temperature=0.7)
    assert result == "Generated text"
    mock_anthropic.Anthropic.return_value.completions.create.assert_called_once_with(
        model="claude-3-haiku-20240307",
        prompt="Test prompt",
        max_tokens_to_sample=10,
        temperature=0.7
    )

def test_generate_with_custom_model(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.return_value.completion = "Custom model text"
    result = anthropic_provider.generate("Test prompt", model="claude-2.1", max_tokens=20, temperature=0.5)
    assert result == "Custom model text"
    mock_anthropic.Anthropic.return_value.completions.create.assert_called_once_with(
        model="claude-2.1",
        prompt="Test prompt",
        max_tokens_to_sample=20,
        temperature=0.5
    )

def test_generate_api_error(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.side_effect = APIError("API Error")
    with pytest.raises(APIError, match="API Error"):
        anthropic_provider.generate("Test prompt")

def test_generate_rate_limit_error(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.side_effect = RateLimitError("Rate limit exceeded")
    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        anthropic_provider.generate("Test prompt")

def test_generate_authentication_error(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.side_effect = AuthenticationError("Invalid API key")
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        anthropic_provider.generate("Test prompt")

def test_generate_stream(anthropic_provider, mock_anthropic):
    mock_stream = Mock()
    mock_stream.__enter__.return_value = iter([
        Mock(completion="Chunk 1"),
        Mock(completion="Chunk 2"),
        Mock(completion="Chunk 3")
    ])
    mock_anthropic.Anthropic.return_value.completions.create.return_value = mock_stream

    result = list(anthropic_provider.generate_stream("Test prompt", max_tokens=30, temperature=0.8))
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_anthropic.Anthropic.return_value.completions.create.assert_called_once_with(
        model="claude-3-haiku-20240307",
        prompt="Test prompt",
        max_tokens_to_sample=30,
        temperature=0.8,
        stream=True
    )

def test_generate_stream_error(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.side_effect = APIError("Stream Error")
    with pytest.raises(APIError, match="Stream Error"):
        list(anthropic_provider.generate_stream("Test prompt"))

def test_embed(anthropic_provider):
    with pytest.raises(NotImplementedError, match="Embedding not currently supported for Anthropic"):
        anthropic_provider.embed("Test text")

def test_chat(anthropic_provider, mock_anthropic):
    mock_response = Mock()
    mock_response.content = [Mock(text="Chat response")]
    mock_response.model = "claude-3-haiku-20240307"
    mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = anthropic_provider.chat(messages, max_tokens=50, temperature=0.7)

    assert result == {
        "role": "assistant",
        "content": "Chat response",
        "model": "claude-3-haiku-20240307",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    }
    mock_anthropic.Anthropic.return_value.messages.create.assert_called_once_with(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        temperature=0.7,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    )

def test_chat_api_error(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = APIError("Chat API Error")
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(APIError, match="Chat API Error"):
        anthropic_provider.chat(messages)

def test_list_available_models(anthropic_provider):
    models = anthropic_provider.list_available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, dict) for model in models)
    assert all("id" in model and "description" in model for model in models)

def test_unsupported_model(anthropic_provider, mock_anthropic):
    with pytest.raises(ValueError, match="Unsupported model"):
        anthropic_provider.generate("Test prompt", model="unsupported-model")

def test_generate_with_system_prompt(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.return_value.completion = "Generated text with system prompt"
    result = anthropic_provider.generate("Test prompt", system_prompt="You are a helpful assistant.")
    assert result == "Generated text with system prompt"
    mock_anthropic.Anthropic.return_value.completions.create.assert_called_once_with(
        model="claude-3-haiku-20240307",
        prompt="Human: You are a helpful assistant.\n\nHuman: Test prompt\n\nAssistant:",
        max_tokens_to_sample=150,
        temperature=0.7
    )

def test_chat_with_system_message(anthropic_provider, mock_anthropic):
    mock_response = Mock()
    mock_response.content = [Mock(text="Chat response with system message")]
    mock_response.model = "claude-3-haiku-20240307"
    mock_response.usage = {"prompt_tokens": 15, "completion_tokens": 25}
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = anthropic_provider.chat(messages)

    assert result["content"] == "Chat response with system message"
    mock_anthropic.Anthropic.return_value.messages.create.assert_called_once()
    call_args = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "You are a helpful assistant."

def test_generate_with_stop_sequences(anthropic_provider, mock_anthropic):
    mock_anthropic.Anthropic.return_value.completions.create.return_value.completion = "Generated text with stop"
    result = anthropic_provider.generate("Test prompt", stop_sequences=["END", "STOP"])
    assert result == "Generated text with stop"
    mock_anthropic.Anthropic.return_value.completions.create.assert_called_once_with(
        model="claude-3-haiku-20240307",
        prompt="Test prompt",
        max_tokens_to_sample=150,
        temperature=0.7,
        stop_sequences=["END", "STOP"]
    )

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])