# tests/unit/test_utils/test_config.py

import pytest
import os
import json
from unittest.mock import patch, mock_open
from utils.config import Config, get_config

@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'GEMINI_API_KEY': 'test_gemini_key',
        'AZURE_OPENAI_API_KEY': 'test_azure_openai_key',
        'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
        'AWS_ACCESS_KEY_ID': 'test_aws_access_key_id',
        'AWS_SECRET_ACCESS_KEY': 'test_aws_secret_access_key',
        'AWS_REGION_NAME': 'us-west-2',
        'OLLAMA_HOST': 'http://localhost:11434',
        'LMSTUDIO_API_BASE': 'http://localhost:1234/v1',
        'LOG_LEVEL': 'DEBUG'
    }):
        yield

@pytest.fixture
def mock_config_file():
    config_data = {
        'openai_api_key': 'file_openai_key',
        'anthropic_api_key': 'file_anthropic_key',
        'custom_setting': 'custom_value'
    }
    with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
        yield

def test_config_initialization(mock_env_vars, mock_config_file):
    config = Config()
    assert config.get('openai_api_key') == 'test_openai_key'
    assert config.get('anthropic_api_key') == 'test_anthropic_key'
    assert config.get('gemini_api_key') == 'test_gemini_key'
    assert config.get('azure_openai_api_key') == 'test_azure_openai_key'
    assert config.get('azure_openai_api_base') == 'https://test.openai.azure.com/'
    assert config.get('aws_access_key_id') == 'test_aws_access_key_id'
    assert config.get('aws_secret_access_key') == 'test_aws_secret_access_key'
    assert config.get('aws_region_name') == 'us-west-2'
    assert config.get('ollama_host') == 'http://localhost:11434'
    assert config.get('lmstudio_api_base') == 'http://localhost:1234/v1'
    assert config.get('log_level') == 'DEBUG'
    assert config.get('custom_setting') == 'custom_value'

def test_config_get_default():
    config = Config()
    assert config.get('non_existent_key', 'default_value') == 'default_value'

def test_config_set():
    config = Config()
    config.set('new_key', 'new_value')
    assert config.get('new_key') == 'new_value'

def test_config_get_all():
    config = Config()
    config.set('key1', 'value1')
    config.set('key2', 'value2')
    all_config = config.get_all()
    assert 'key1' in all_config
    assert 'key2' in all_config
    assert all_config['key1'] == 'value1'
    assert all_config['key2'] == 'value2'

@patch('json.dump')
def test_config_save_to_file(mock_json_dump):
    config = Config()
    config.set('key1', 'value1')
    config.set('key2', 'value2')
    
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        config.save_to_file('test_config.json')
    
    mock_file.assert_called_once_with(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'test_config.json'), 'w')
    mock_json_dump.assert_called_once()
    args, _ = mock_json_dump.call_args
    saved_config = args[0]
    assert 'key1' in saved_config
    assert 'key2' in saved_config

def test_get_config_singleton():
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2

@patch('utils.config.Config')
def test_get_config_initialization(mock_config_class):
    get_config()
    mock_config_class.assert_called_once()

def test_config_priority(mock_env_vars, mock_config_file):
    config = Config()
    # Environment variable should take precedence over file
    assert config.get('openai_api_key') == 'test_openai_key'
    # File value should be used when no environment variable is set
    assert config.get('custom_setting') == 'custom_value'

@patch.dict(os.environ, {'NEW_ENV_VAR': 'new_env_value'})
def test_config_load_new_env_var():
    config = Config()
    assert config.get('NEW_ENV_VAR') == 'new_env_value'

def test_config_non_existent_file():
    with patch('os.path.exists', return_value=False):
        config = Config()
        assert config.get('custom_setting', 'default') == 'default'

def test_config_file_load_error():
    with patch('builtins.open', side_effect=IOError("File read error")):
        config = Config()
        assert config.get('custom_setting', 'default') == 'default'

def test_config_set_and_save(mock_env_vars):
    config = Config()
    config.set('new_setting', 'new_value')
    assert config.get('new_setting') == 'new_value'
    
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        config.save_to_file('test_config.json')
    
    mock_file.assert_called_once()
    written_data = mock_file().write.call_args[0][0]
    saved_config = json.loads(written_data)
    assert 'new_setting' in saved_config
    assert saved_config['new_setting'] == 'new_value'

def test_config_update_existing_value():
    config = Config()
    config.set('existing_key', 'old_value')
    config.set('existing_key', 'new_value')
    assert config.get('existing_key') == 'new_value'

def test_config_case_sensitivity():
    config = Config()
    config.set('CaseSensitiveKey', 'Value1')
    config.set('casesensitivekey', 'Value2')
    assert config.get('CaseSensitiveKey') == 'Value1'
    assert config.get('casesensitivekey') == 'Value2'

@patch('utils.config.logger.info')
def test_config_logging(mock_logger_info):
    Config()
    mock_logger_info.assert_called_with("Loaded configuration from environment variables")

@patch('utils.config.logger.error')
def test_config_error_logging(mock_logger_error):
    with patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0)):
        Config()
    mock_logger_error.assert_called()

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])