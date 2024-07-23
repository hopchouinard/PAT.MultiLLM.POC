# tests/unit/test_providers/test_gemini_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.gemini_provider import GeminiProvider
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse, Content, SafetyRating

@pytest.fixture
def mock_genai():
    with patch('llm_providers.gemini_provider.genai') as mock:
        yield mock

@pytest.fixture
def gemini_provider(mock_genai, mock_config):
    return GeminiProvider()

def test_gemini_provider_initialization(gemini_provider, mock_config, mock_genai):
    assert isinstance(gemini_provider, GeminiProvider)
    assert mock_genai.configure.called_with(api_key='test_gemini_key')

def create_mock_response(text):
    mock_response = Mock(spec=GenerateContentResponse)
    mock_response.text = text
    mock_response.prompt_feedback = None
    mock_response.candidates = [
        Content(
            parts=[{"text": text}],
            role="model",
            content_type="text",
            safety_ratings=[SafetyRating(category="HARM_CATEGORY_UNSPECIFIED", probability="NEGLIGIBLE")]
        )
    ]
    return mock_response

def test_generate(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.return_value = create_mock_response("Generated text")
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_provider.generate("Test prompt", model="gemini-1.5-flash", max_tokens=10, temperature=0.7)
    assert result == "Generated text"
    mock_model.generate_content.assert_called_once()
    assert mock_model.generate_content.call_args[0][0] == "Test prompt"
    assert mock_model.generate_content.call_args[1]['generation_config'].temperature == 0.7

def test_generate_with_custom_model(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.return_value = create_mock_response("Custom model text")
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_provider.generate("Test prompt", model="gemini-pro", max_tokens=20, temperature=0.5)
    assert result == "Custom model text"
    mock_genai.GenerativeModel.assert_called_once_with("gemini-pro")
    mock_model.generate_content.assert_called_once()

def test_generate_api_error(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.side_effect = Exception("API Error")
    mock_genai.GenerativeModel.return_value = mock_model

    with pytest.raises(Exception, match="API Error"):
        gemini_provider.generate("Test prompt")

def test_generate_stream(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.return_value = [
        create_mock_response("Chunk 1"),
        create_mock_response("Chunk 2"),
        create_mock_response("Chunk 3")
    ]
    mock_genai.GenerativeModel.return_value = mock_model

    result = list(gemini_provider.generate_stream("Test prompt", model="gemini-1.5-flash", max_tokens=30, temperature=0.8))
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_model.generate_content.assert_called_once()
    assert mock_model.generate_content.call_args[1]['stream'] == True

def test_generate_stream_error(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.side_effect = Exception("Stream Error")
    mock_genai.GenerativeModel.return_value = mock_model

    with pytest.raises(Exception, match="Stream Error"):
        list(gemini_provider.generate_stream("Test prompt"))

def test_embed(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_genai.get_model.return_value = mock_model

    result = gemini_provider.embed("Test text", model="embedding-001")
    assert result == [0.1, 0.2, 0.3]
    mock_genai.get_model.assert_called_once_with("embedding-001")
    mock_model.embed_content.assert_called_once_with("Test text")

def test_embed_error(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.embed_content.side_effect = Exception("Embedding Error")
    mock_genai.get_model.return_value = mock_model

    with pytest.raises(Exception, match="Embedding Error"):
        gemini_provider.embed("Test text")

def test_chat(gemini_provider, mock_genai):
    mock_chat = Mock()
    mock_chat.send_message.return_value = create_mock_response("Chat response")
    mock_model = Mock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai.GenerativeModel.return_value = mock_model

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "model", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = gemini_provider.chat(messages, model="gemini-1.5-flash", max_tokens=50, temperature=0.7)

    assert result == {
        "role": "model",
        "content": "Chat response",
        "model": "gemini-1.5-flash",
    }
    mock_genai.GenerativeModel.assert_called_once_with("gemini-1.5-flash")
    mock_model.start_chat.assert_called_once_with(history=[])
    assert mock_chat.send_message.call_count == 3

def test_chat_api_error(gemini_provider, mock_genai):
    mock_chat = Mock()
    mock_chat.send_message.side_effect = Exception("Chat API Error")
    mock_model = Mock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai.GenerativeModel.return_value = mock_model

    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(Exception, match="Chat API Error"):
        gemini_provider.chat(messages)

def test_list_available_models(gemini_provider, mock_genai):
    mock_genai.list_models.return_value = [
        Mock(name="gemini-1.5-flash", description="Fast model"),
        Mock(name="gemini-pro", description="Powerful model")
    ]
    models = gemini_provider.list_available_models()
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0]["name"] == "gemini-1.5-flash"
    assert models[1]["name"] == "gemini-pro"
    assert "description" in models[0] and "description" in models[1]

def test_set_safety_settings(gemini_provider, mock_genai):
    gemini_provider.set_safety_settings(
        harassment=genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE,
        hate_speech=genai.types.HarmBlockThreshold.HIGH_AND_ABOVE
    )
    mock_genai.configure.assert_called_with(safety_settings=[
        {'category': genai.types.HarmCategory.HARASSMENT, 'threshold': genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE},
        {'category': genai.types.HarmCategory.HATE_SPEECH, 'threshold': genai.types.HarmBlockThreshold.HIGH_AND_ABOVE}
    ])

def test_generate_with_safety_settings(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.return_value = create_mock_response("Safe generated text")
    mock_genai.GenerativeModel.return_value = mock_model

    gemini_provider.set_safety_settings(
        dangerous_content=genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
    )
    
    result = gemini_provider.generate("Test prompt")
    assert result == "Safe generated text"
    mock_model.generate_content.assert_called_once()
    safety_settings = mock_genai.configure.call_args[1]['safety_settings']
    assert any(setting['category'] == genai.types.HarmCategory.DANGEROUS_CONTENT for setting in safety_settings)

def test_generate_with_custom_generation_config(gemini_provider, mock_genai):
    mock_model = Mock()
    mock_model.generate_content.return_value = create_mock_response("Custom config generated text")
    mock_genai.GenerativeModel.return_value = mock_model

    result = gemini_provider.generate(
        "Test prompt",
        temperature=0.9,
        top_p=0.95,
        top_k=40,
        max_output_tokens=100
    )
    assert result == "Custom config generated text"
    mock_model.generate_content.assert_called_once()
    generation_config = mock_model.generate_content.call_args[1]['generation_config']
    assert generation_config.temperature == 0.9
    assert generation_config.top_p == 0.95
    assert generation_config.top_k == 40
    assert generation_config.max_output_tokens == 100

def test_chat_with_history(gemini_provider, mock_genai):
    mock_chat = Mock()
    mock_chat.send_message.return_value = create_mock_response("Chat response with history")
    mock_model = Mock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai.GenerativeModel.return_value = mock_model

    messages = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "model", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What's a famous landmark there?"}
    ]
    result = gemini_provider.chat(messages)

    assert result["content"] == "Chat response with history"
    mock_model.start_chat.assert_called_once()
    assert mock_chat.send_message.call_count == 3
    assert mock_chat.history == [
        {"role": "user", "parts": ["What's the capital of France?"]},
        {"role": "model", "parts": ["The capital of France is Paris."]},
        {"role": "user", "parts": ["What's a famous landmark there?"]}
    ]

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])