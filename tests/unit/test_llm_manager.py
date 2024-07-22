# tests/unit/test_llm_manager.py

import pytest
from unittest.mock import Mock, patch
from llm_manager import LLMManager
from llm_providers import LLMProvider

@pytest.fixture
def llm_manager(mock_openai_provider, mock_anthropic_provider, mock_gemini_provider, 
                mock_ollama_provider, mock_lmstudio_provider, mock_azure_openai_provider, 
                mock_aws_bedrock_provider):
    # This fixture is defined in conftest.py
    return llm_manager

def test_llm_manager_initialization(llm_manager):
    assert isinstance(llm_manager, LLMManager)
    assert set(llm_manager.providers.keys()) == {'openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'}
    for provider in llm_manager.providers.values():
        assert isinstance(provider, LLMProvider)

def test_generate_single_provider(llm_manager, sample_text):
    result = llm_manager.generate(sample_text, provider='openai')
    assert 'openai' in result
    assert result['openai']['status'] == 'success'
    assert result['openai']['result'] == "OpenAI generated text"

def test_generate_all_providers(llm_manager, sample_text):
    result = llm_manager.generate(sample_text)
    assert set(result.keys()) == {'openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'}
    for provider_result in result.values():
        assert provider_result['status'] == 'success'
        assert 'result' in provider_result

def test_generate_with_error(llm_manager, sample_text):
    llm_manager.providers['openai'].generate.side_effect = Exception("API Error")
    result = llm_manager.generate(sample_text, provider='openai')
    assert result['openai']['status'] == 'error'
    assert "API Error" in result['openai']['error']

def test_generate_with_invalid_provider(llm_manager, sample_text):
    with pytest.raises(ValueError):
        llm_manager.generate(sample_text, provider='invalid_provider')

def test_generate_stream(llm_manager, sample_text):
    chunks = list(llm_manager.generate_stream(sample_text, provider='anthropic'))
    assert len(chunks) == 3
    assert all(chunk['status'] == 'success' for chunk in chunks)
    assert ''.join(chunk['chunk'] for chunk in chunks) == "Anthropic streamed text"

def test_generate_stream_with_error(llm_manager, sample_text):
    llm_manager.providers['anthropic'].generate_stream.side_effect = Exception("Stream Error")
    chunks = list(llm_manager.generate_stream(sample_text, provider='anthropic'))
    assert len(chunks) == 1
    assert chunks[0]['status'] == 'error'
    assert "Stream Error" in chunks[0]['error']

def test_embed_single_provider(llm_manager, sample_text):
    result = llm_manager.embed(sample_text, provider='gemini')
    assert 'gemini' in result
    assert result['gemini']['status'] == 'success'
    assert result['gemini']['result'] == [0.7, 0.8, 0.9]

def test_embed_all_providers(llm_manager, sample_text):
    result = llm_manager.embed(sample_text)
    assert set(result.keys()) == {'openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'}
    for provider_result in result.values():
        assert provider_result['status'] == 'success'
        assert isinstance(provider_result['result'], list)

def test_embed_with_error(llm_manager, sample_text):
    llm_manager.providers['gemini'].embed.side_effect = Exception("Embedding Error")
    result = llm_manager.embed(sample_text, provider='gemini')
    assert result['gemini']['status'] == 'error'
    assert "Embedding Error" in result['gemini']['error']

def test_chat_single_provider(llm_manager, sample_chat_messages):
    result = llm_manager.chat(sample_chat_messages, provider='ollama')
    assert 'ollama' in result
    assert result['ollama']['status'] == 'success'
    assert result['ollama']['result']['content'] == "Ollama chat response"

def test_chat_all_providers(llm_manager, sample_chat_messages):
    result = llm_manager.chat(sample_chat_messages)
    assert set(result.keys()) == {'openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'}
    for provider_result in result.values():
        assert provider_result['status'] == 'success'
        assert 'content' in provider_result['result']

def test_chat_with_error(llm_manager, sample_chat_messages):
    llm_manager.providers['ollama'].chat.side_effect = Exception("Chat Error")
    result = llm_manager.chat(sample_chat_messages, provider='ollama')
    assert result['ollama']['status'] == 'error'
    assert "Chat Error" in result['ollama']['error']

def test_generate_parallel(llm_manager):
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    result = llm_manager.generate_parallel(prompts)
    assert len(result) == 3
    for prompt_result in result.values():
        assert set(prompt_result.keys()) == {'openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'}
        for provider_result in prompt_result.values():
            assert provider_result['status'] == 'success'
            assert 'result' in provider_result

def test_generate_parallel_with_error(llm_manager):
    prompts = ["Prompt 1", "Prompt 2"]
    llm_manager.providers['openai'].generate.side_effect = [Exception("API Error"), "OpenAI generated text"]
    result = llm_manager.generate_parallel(prompts)
    assert len(result) == 2
    assert result[0]['openai']['status'] == 'error'
    assert "API Error" in result[0]['openai']['error']
    assert result[1]['openai']['status'] == 'success'
    assert result[1]['openai']['result'] == "OpenAI generated text"

def test_get_provider_info(llm_manager):
    info = llm_manager.get_provider_info()
    assert set(info.keys()) == {'openai', 'anthropic', 'gemini', 'ollama', 'lmstudio', 'azure_openai', 'aws_bedrock'}
    for provider_info in info.values():
        assert 'name' in provider_info
        assert 'type' in provider_info
        assert 'methods' in provider_info
        assert 'available_models' in provider_info

def test_get_provider_info_single_provider(llm_manager):
    info = llm_manager.get_provider_info(provider='lmstudio')
    assert 'lmstudio' in info
    assert info['lmstudio']['name'] == 'lmstudio'
    assert info['lmstudio']['type'] == 'LMStudioProvider'
    assert 'generate' in info['lmstudio']['methods']
    assert 'embed' in info['lmstudio']['methods']
    assert 'chat' in info['lmstudio']['methods']

def test_get_provider_info_with_error(llm_manager):
    llm_manager.providers['azure_openai'].list_available_models.side_effect = Exception("Info Error")
    info = llm_manager.get_provider_info(provider='azure_openai')
    assert 'azure_openai' in info
    assert 'error' in info['azure_openai']
    assert "Info Error" in info['azure_openai']['error']

# Additional tests for edge cases and specific scenarios

def test_generate_with_custom_parameters(llm_manager, sample_text):
    result = llm_manager.generate(sample_text, provider='openai', max_tokens=50, temperature=0.8)
    assert result['openai']['status'] == 'success'
    llm_manager.providers['openai'].generate.assert_called_with(sample_text, max_tokens=50, temperature=0.8)

def test_chat_with_custom_parameters(llm_manager, sample_chat_messages):
    result = llm_manager.chat(sample_chat_messages, provider='anthropic', max_tokens=100, temperature=0.5)
    assert result['anthropic']['status'] == 'success'
    llm_manager.providers['anthropic'].chat.assert_called_with(sample_chat_messages, max_tokens=100, temperature=0.5)

def test_generate_stream_interruption(llm_manager, sample_text):
    def mock_stream():
        yield "First chunk"
        raise Exception("Stream interrupted")
        yield "This should not be reached"

    llm_manager.providers['gemini'].generate_stream.return_value = mock_stream()
    chunks = list(llm_manager.generate_stream(sample_text, provider='gemini'))
    assert len(chunks) == 2
    assert chunks[0]['status'] == 'success' and chunks[0]['chunk'] == "First chunk"
    assert chunks[1]['status'] == 'error' and "Stream interrupted" in chunks[1]['error']

def test_embed_with_different_dimensions(llm_manager, sample_text):
    llm_manager.providers['openai'].embed.return_value = [0.1, 0.2, 0.3]
    llm_manager.providers['anthropic'].embed.return_value = [0.4, 0.5, 0.6, 0.7]
    result = llm_manager.embed(sample_text, provider=['openai', 'anthropic'])
    assert len(result['openai']['result']) == 3
    assert len(result['anthropic']['result']) == 4

def test_generate_parallel_with_timeout(llm_manager):
    import time

    def slow_generate(prompt):
        time.sleep(2)
        return "Slow response"

    llm_manager.providers['aws_bedrock'].generate.side_effect = slow_generate
    prompts = ["Fast prompt", "Slow prompt"]
    
    with patch('llm_manager.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [
            Mock(result=lambda: {"openai": {"status": "success", "result": "Fast response"}}),
            Mock(side_effect=TimeoutError("Operation timed out"))
        ]
        result = llm_manager.generate_parallel(prompts, timeout=1)
    
    assert result[0]['openai']['status'] == 'success'
    assert result[0]['openai']['result'] == "Fast response"
    assert result[1]['openai']['status'] == 'error'
    assert "Operation timed out" in result[1]['openai']['error']

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])