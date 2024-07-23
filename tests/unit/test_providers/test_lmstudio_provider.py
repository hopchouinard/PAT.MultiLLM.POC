# tests/unit/test_providers/test_lmstudio_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.lmstudio_provider import LMStudioProvider
import openai

@pytest.fixture
def mock_openai():
    with patch('llm_providers.lmstudio_provider.openai') as mock:
        yield mock

@pytest.fixture
def lmstudio_provider(mock_openai, mock_config):
    return LMStudioProvider()

def test_lmstudio_provider_initialization(lmstudio_provider, mock_config, mock_openai):
    assert isinstance(lmstudio_provider, LMStudioProvider)
    assert lmstudio_provider.api_base == 'http://localhost:1234/v1'
    mock_openai.OpenAI.assert_called_once_with(base_url='http://localhost:1234/v1', api_key="not-needed")

def test_generate(lmstudio_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].message.content = "Generated text"
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

    result = lmstudio_provider.generate("Test prompt", model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", max_tokens=10, temperature=0.7)
    assert result == "Generated text"
    mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        temperature=0.7
    )

def test_generate_with_custom_model(lmstudio_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].message.content = "Custom model text"
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

    result = lmstudio_provider.generate("Test prompt", model="custom-model", max_tokens=20, temperature=0.5)
    assert result == "Custom model text"
    mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="custom-model",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=20,
        temperature=0.5
    )

def test_generate_api_error(lmstudio_provider, mock_openai):
    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = openai.OpenAIError("API Error")
    with pytest.raises(openai.OpenAIError, match="API Error"):
        lmstudio_provider.generate("Test prompt")

def test_generate_stream(lmstudio_provider, mock_openai):
    mock_stream = Mock()
    mock_stream.__iter__.return_value = [
        Mock(choices=[Mock(delta=Mock(content="Chunk 1"))]),
        Mock(choices=[Mock(delta=Mock(content="Chunk 2"))]),
        Mock(choices=[Mock(delta=Mock(content="Chunk 3"))])
    ]
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_stream

    result = list(lmstudio_provider.generate_stream("Test prompt", model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", max_tokens=30, temperature=0.8))
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=30,
        temperature=0.8,
        stream=True
    )

def test_generate_stream_error(lmstudio_provider, mock_openai):
    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = openai.OpenAIError("Stream Error")
    with pytest.raises(openai.OpenAIError, match="Stream Error"):
        list(lmstudio_provider.generate_stream("Test prompt"))

def test_embed(lmstudio_provider, mock_openai):
    mock_embedding = Mock()
    mock_embedding.data[0].embedding = [0.1, 0.2, 0.3]
    mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_embedding

    result = lmstudio_provider.embed("Test text", model="nomic-embed-text-v1.5")
    assert result == [0.1, 0.2, 0.3]
    mock_openai.OpenAI.return_value.embeddings.create.assert_called_once_with(
        input=["Test text"],
        model="nomic-embed-text-v1.5"
    )

def test_embed_error(lmstudio_provider, mock_openai):
    mock_openai.OpenAI.return_value.embeddings.create.side_effect = openai.OpenAIError("Embedding Error")
    with pytest.raises(openai.OpenAIError, match="Embedding Error"):
        lmstudio_provider.embed("Test text")

def test_chat(lmstudio_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].message.content = "Chat response"
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = lmstudio_provider.chat(messages, model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", max_tokens=50, temperature=0.7)

    assert result == {
        "role": "assistant",
        "content": "Chat response",
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    }
    mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        messages=messages,
        max_tokens=50,
        temperature=0.7
    )

def test_chat_api_error(lmstudio_provider, mock_openai):
    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = openai.OpenAIError("Chat API Error")
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(openai.OpenAIError, match="Chat API Error"):
        lmstudio_provider.chat(messages)

def test_list_available_models(lmstudio_provider):
    models = lmstudio_provider.list_available_models()
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0]["id"] == "local-model"
    assert models[1]["id"] == "local-embedding-model"

def test_get_model_info(lmstudio_provider):
    info = lmstudio_provider.get_model_info("local-model")
    assert info["id"] == "local-model"
    assert info["name"] == "Local LM Studio Model"
    assert "description" in info

def test_set_model_parameters(lmstudio_provider):
    # This method is a placeholder in LMStudioProvider
    lmstudio_provider.set_model_parameters(param1="value1", param2="value2")
    # Ensure it doesn't raise any exception
    assert True

def test_generate_with_system_message(lmstudio_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].message.content = "Generated with system message"
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

    result = lmstudio_provider.generate(
        "Test prompt",
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        system_message="You are a helpful assistant."
    )
    assert result == "Generated with system message"
    mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test prompt"}
        ],
        max_tokens=150,
        temperature=0.7
    )

def test_generate_with_custom_parameters(lmstudio_provider, mock_openai):
    mock_completion = Mock()
    mock_completion.choices[0].message.content = "Generated with custom parameters"
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

    result = lmstudio_provider.generate(
        "Test prompt",
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )
    assert result == "Generated with custom parameters"
    mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])