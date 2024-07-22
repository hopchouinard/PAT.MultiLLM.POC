# tests/unit/test_providers/test_openai_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.openai_provider import OpenAIProvider
import openai
from openai import OpenAIError, APIError, RateLimitError, AuthenticationError

@pytest.fixture
def mock_openai():
    with patch('llm_providers.openai_provider.openai') as mock:
        yield mock

@pytest.fixture
def openai_provider(mock_openai, mock_config):
    return OpenAIProvider()

def test_openai_provider_initialization(openai_provider, mock_config):
    assert isinstance(openai_provider, OpenAIProvider)
    assert openai_provider.api_key == 'test_openai_key'
    assert openai.api_key == 'test_openai_key'

def test_generate_chat_model(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.return_value.choices[0].message.content = "Generated chat text"
    result = openai_provider.generate("Test prompt", model="gpt-4o-mini", max_tokens=10, temperature=0.7)
    assert result == "Generated chat text"
    mock_openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        temperature=0.7
    )

def test_generate_completion_model(openai_provider, mock_openai):
    mock_openai.Completion.create.return_value.choices[0].text = "Generated completion text"
    result = openai_provider.generate("Test prompt", model="text-davinci-003", max_tokens=20, temperature=0.5)
    assert result == "Generated completion text"
    mock_openai.Completion.create.assert_called_once_with(
        model="text-davinci-003",
        prompt="Test prompt",
        max_tokens=20,
        temperature=0.5
    )

def test_generate_api_error(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.side_effect = APIError("API Error")
    with pytest.raises(APIError, match="API Error"):
        openai_provider.generate("Test prompt", model="gpt-4o-mini")

def test_generate_rate_limit_error(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.side_effect = RateLimitError("Rate limit exceeded")
    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        openai_provider.generate("Test prompt", model="gpt-4o-mini")

def test_generate_authentication_error(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.side_effect = AuthenticationError("Invalid API key")
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        openai_provider.generate("Test prompt", model="gpt-4o-mini")

def test_generate_stream_chat_model(openai_provider, mock_openai):
    mock_stream = Mock()
    mock_stream.__iter__.return_value = [
        Mock(choices=[Mock(delta=Mock(content="Chunk 1"))]),
        Mock(choices=[Mock(delta=Mock(content="Chunk 2"))]),
        Mock(choices=[Mock(delta=Mock(content="Chunk 3"))])
    ]
    mock_openai.ChatCompletion.create.return_value = mock_stream

    result = list(openai_provider.generate_stream("Test prompt", model="gpt-4o-mini", max_tokens=30, temperature=0.8))
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=30,
        temperature=0.8,
        stream=True
    )

def test_generate_stream_completion_model(openai_provider, mock_openai):
    mock_stream = Mock()
    mock_stream.__iter__.return_value = [
        Mock(choices=[Mock(text="Chunk A")]),
        Mock(choices=[Mock(text="Chunk B")]),
        Mock(choices=[Mock(text="Chunk C")])
    ]
    mock_openai.Completion.create.return_value = mock_stream

    result = list(openai_provider.generate_stream("Test prompt", model="text-davinci-003", max_tokens=30, temperature=0.8))
    assert result == ["Chunk A", "Chunk B", "Chunk C"]
    mock_openai.Completion.create.assert_called_once_with(
        model="text-davinci-003",
        prompt="Test prompt",
        max_tokens=30,
        temperature=0.8,
        stream=True
    )

def test_generate_stream_error(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.side_effect = APIError("Stream Error")
    with pytest.raises(APIError, match="Stream Error"):
        list(openai_provider.generate_stream("Test prompt", model="gpt-4o-mini"))

def test_embed(openai_provider, mock_openai):
    mock_openai.Embedding.create.return_value = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
    result = openai_provider.embed("Test text", model="text-embedding-ada-002")
    assert result == [0.1, 0.2, 0.3]
    mock_openai.Embedding.create.assert_called_once_with(
        input=["Test text"],
        model="text-embedding-ada-002"
    )

def test_embed_error(openai_provider, mock_openai):
    mock_openai.Embedding.create.side_effect = APIError("Embedding Error")
    with pytest.raises(APIError, match="Embedding Error"):
        openai_provider.embed("Test text")

def test_chat(openai_provider, mock_openai):
    mock_response = Mock()
    mock_response.choices[0].message.content = "Chat response"
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    mock_openai.ChatCompletion.create.return_value = mock_response

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = openai_provider.chat(messages, model="gpt-4o-mini", max_tokens=50, temperature=0.7)

    assert result == {
        "role": "assistant",
        "content": "Chat response",
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
    mock_openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=50,
        temperature=0.7
    )

def test_chat_api_error(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.side_effect = APIError("Chat API Error")
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(APIError, match="Chat API Error"):
        openai_provider.chat(messages)

def test_list_available_models(openai_provider, mock_openai):
    mock_openai.Model.list.return_value.data = [
        Mock(id="gpt-4o-mini", created=1623123456),
        Mock(id="text-davinci-003", created=1622123456)
    ]
    models = openai_provider.list_available_models()
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0]["id"] == "gpt-4o-mini"
    assert models[1]["id"] == "text-davinci-003"
    assert "created" in models[0] and "created" in models[1]

def test_get_model_info(openai_provider, mock_openai):
    mock_openai.Model.retrieve.return_value = Mock(
        id="gpt-4o-mini",
        created=1623123456,
        owned_by="openai"
    )
    info = openai_provider.get_model_info("gpt-4o-mini")
    assert info["id"] == "gpt-4o-mini"
    assert "created" in info
    assert info["owned_by"] == "openai"
    mock_openai.Model.retrieve.assert_called_once_with("gpt-4o-mini")

def test_get_model_info_error(openai_provider, mock_openai):
    mock_openai.Model.retrieve.side_effect = APIError("Model not found")
    with pytest.raises(APIError, match="Model not found"):
        openai_provider.get_model_info("nonexistent-model")

def test_generate_with_custom_parameters(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.return_value.choices[0].message.content = "Custom parameters response"
    result = openai_provider.generate(
        "Test prompt",
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )
    assert result == "Custom parameters response"
    mock_openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )

def test_generate_with_system_message(openai_provider, mock_openai):
    mock_openai.ChatCompletion.create.return_value.choices[0].message.content = "Response with system message"
    result = openai_provider.generate(
        "Test prompt",
        model="gpt-4o-mini",
        system_message="You are a helpful assistant."
    )
    assert result == "Response with system message"
    mock_openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test prompt"}
        ],
        max_tokens=150,
        temperature=0.7
    )

def test_chat_with_functions(openai_provider, mock_openai):
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    ]
    
    mock_response = Mock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.function_call = {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}'
    }
    mock_response.model = "gpt-4o-mini"
    mock_openai.ChatCompletion.create.return_value = mock_response

    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    result = openai_provider.chat(messages, model="gpt-4o-mini", functions=functions)

    assert result["role"] == "assistant"
    assert result["content"] is None
    assert result["function_call"] == {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}'
    }
    mock_openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        max_tokens=150,
        temperature=0.7
    )

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])