# tests/unit/test_providers/test_ollama_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.ollama_provider import OllamaProvider
import requests

@pytest.fixture
def mock_ollama():
    with patch('llm_providers.ollama_provider.ollama') as mock:
        yield mock

@pytest.fixture
def ollama_provider(mock_ollama, mock_config):
    return OllamaProvider()

def test_ollama_provider_initialization(ollama_provider, mock_config, mock_ollama):
    assert isinstance(ollama_provider, OllamaProvider)
    assert ollama_provider.host == 'http://localhost:11434'
    mock_ollama.set_host.assert_called_once_with('http://localhost:11434')

def test_generate(ollama_provider, mock_ollama):
    mock_ollama.generate.return_value = {'response': 'Generated text'}
    result = ollama_provider.generate("Test prompt", model="llama3", max_tokens=10, temperature=0.7)
    assert result == 'Generated text'
    mock_ollama.generate.assert_called_once_with(
        model="llama3",
        prompt="Test prompt",
        options={
            "num_predict": 10,
            "temperature": 0.7
        }
    )

def test_generate_with_custom_model(ollama_provider, mock_ollama):
    mock_ollama.generate.return_value = {'response': 'Custom model text'}
    result = ollama_provider.generate("Test prompt", model="custom-model", max_tokens=20, temperature=0.5)
    assert result == 'Custom model text'
    mock_ollama.generate.assert_called_once_with(
        model="custom-model",
        prompt="Test prompt",
        options={
            "num_predict": 20,
            "temperature": 0.5
        }
    )

def test_generate_api_error(ollama_provider, mock_ollama):
    mock_ollama.generate.side_effect = requests.RequestException("API Error")
    with pytest.raises(requests.RequestException, match="API Error"):
        ollama_provider.generate("Test prompt")

def test_generate_stream(ollama_provider, mock_ollama):
    mock_ollama.generate.return_value = iter([
        {'response': 'Chunk 1'},
        {'response': 'Chunk 2'},
        {'response': 'Chunk 3'}
    ])
    result = list(ollama_provider.generate_stream("Test prompt", model="llama3", max_tokens=30, temperature=0.8))
    assert result == ['Chunk 1', 'Chunk 2', 'Chunk 3']
    mock_ollama.generate.assert_called_once_with(
        model="llama3",
        prompt="Test prompt",
        options={
            "num_predict": 30,
            "temperature": 0.8
        },
        stream=True
    )

def test_generate_stream_error(ollama_provider, mock_ollama):
    mock_ollama.generate.side_effect = requests.RequestException("Stream Error")
    with pytest.raises(requests.RequestException, match="Stream Error"):
        list(ollama_provider.generate_stream("Test prompt"))

def test_embed(ollama_provider, mock_ollama):
    mock_ollama.embeddings.return_value = {'embedding': [0.1, 0.2, 0.3]}
    result = ollama_provider.embed("Test text", model="mxbai-embed-large")
    assert result == [0.1, 0.2, 0.3]
    mock_ollama.embeddings.assert_called_once_with(model="mxbai-embed-large", prompt="Test text")

def test_embed_error(ollama_provider, mock_ollama):
    mock_ollama.embeddings.side_effect = requests.RequestException("Embedding Error")
    with pytest.raises(requests.RequestException, match="Embedding Error"):
        ollama_provider.embed("Test text")

def test_chat(ollama_provider, mock_ollama):
    mock_ollama.chat.return_value = {'message': {'content': 'Chat response'}}
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = ollama_provider.chat(messages, model="llama3", max_tokens=50, temperature=0.7)
    assert result == {
        "role": "assistant",
        "content": "Chat response",
        "model": "llama3",
    }
    mock_ollama.chat.assert_called_once_with(
        model="llama3",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ],
        options={
            "num_predict": 50,
            "temperature": 0.7
        }
    )

def test_chat_api_error(ollama_provider, mock_ollama):
    mock_ollama.chat.side_effect = requests.RequestException("Chat API Error")
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(requests.RequestException, match="Chat API Error"):
        ollama_provider.chat(messages)

def test_list_available_models(ollama_provider, mock_ollama):
    mock_ollama.list.return_value = {
        'models': [
            {'name': 'llama3', 'modified_at': '2023-01-01T00:00:00Z'},
            {'name': 'custom-model', 'modified_at': '2023-01-02T00:00:00Z'}
        ]
    }
    models = ollama_provider.list_available_models()
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0]["name"] == "llama3"
    assert models[1]["name"] == "custom-model"
    assert "modified_at" in models[0] and "modified_at" in models[1]

def test_pull_model(ollama_provider, mock_ollama):
    ollama_provider.pull_model("new-model")
    mock_ollama.pull.assert_called_once_with("new-model")

def test_pull_model_error(ollama_provider, mock_ollama):
    mock_ollama.pull.side_effect = requests.RequestException("Pull Error")
    with pytest.raises(requests.RequestException, match="Pull Error"):
        ollama_provider.pull_model("new-model")

def test_delete_model(ollama_provider, mock_ollama):
    ollama_provider.delete_model("old-model")
    mock_ollama.delete.assert_called_once_with("old-model")

def test_delete_model_error(ollama_provider, mock_ollama):
    mock_ollama.delete.side_effect = requests.RequestException("Delete Error")
    with pytest.raises(requests.RequestException, match="Delete Error"):
        ollama_provider.delete_model("old-model")

def test_generate_with_custom_options(ollama_provider, mock_ollama):
    mock_ollama.generate.return_value = {'response': 'Generated with custom options'}
    result = ollama_provider.generate(
        "Test prompt",
        model="llama3",
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
        repeat_penalty=1.1
    )
    assert result == 'Generated with custom options'
    mock_ollama.generate.assert_called_once_with(
        model="llama3",
        prompt="Test prompt",
        options={
            "num_predict": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    )

def test_chat_with_system_prompt(ollama_provider, mock_ollama):
    mock_ollama.chat.return_value = {'message': {'content': 'Chat response with system prompt'}}
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    result = ollama_provider.chat(messages, model="llama3")
    assert result["content"] == "Chat response with system prompt"
    mock_ollama.chat.assert_called_once()
    call_args = mock_ollama.chat.call_args[1]
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "You are a helpful assistant."

def test_generate_with_stop_sequences(ollama_provider, mock_ollama):
    mock_ollama.generate.return_value = {'response': 'Generated text with stop'}
    result = ollama_provider.generate(
        "Test prompt",
        model="llama3",
        stop=["END", "STOP"]
    )
    assert result == 'Generated text with stop'
    mock_ollama.generate.assert_called_once_with(
        model="llama3",
        prompt="Test prompt",
        options={
            "num_predict": 150,
            "temperature": 0.7,
            "stop": ["END", "STOP"]
        }
    )

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])