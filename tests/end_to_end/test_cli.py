# tests/end_to_end/test_cli.py

import pytest
import subprocess
import os
import json
import tempfile
from unittest.mock import patch

# Helper function to run CLI commands
def run_cli_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode

# Fixture to set up test environment variables
@pytest.fixture
def set_test_env():
    original_env = os.environ.copy()
    os.environ['OPENAI_API_KEY'] = 'test_openai_key'
    os.environ['ANTHROPIC_API_KEY'] = 'test_anthropic_key'
    os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
    os.environ['AZURE_OPENAI_API_KEY'] = 'test_azure_openai_key'
    os.environ['AZURE_OPENAI_API_BASE'] = 'https://test.openai.azure.com/'
    os.environ['AWS_ACCESS_KEY_ID'] = 'test_aws_access_key_id'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'test_aws_secret_access_key'
    os.environ['AWS_REGION_NAME'] = 'us-west-2'
    yield
    os.environ.clear()
    os.environ.update(original_env)

# Test basic generation command
def test_generate_command(set_test_env):
    command = "python main.py --providers openai --task generate --input 'What is Python?' --max_tokens 50"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'Python' in stdout
    assert 'programming language' in stdout.lower()

# Test embedding command
def test_embed_command(set_test_env):
    command = "python main.py --providers openai --task embed --input 'Test embedding' --output embed_result.json"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert os.path.exists('embed_result.json')
    
    with open('embed_result.json', 'r') as f:
        result = json.load(f)
    
    assert 'openai' in result
    assert 'result' in result['openai']
    assert isinstance(result['openai']['result'], list)
    
    os.remove('embed_result.json')

# Test chat command
def test_chat_command(set_test_env):
    command = "python main.py --providers anthropic --task chat --input 'Who won the World Cup in 2022?' --max_tokens 50"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'Argentina' in stdout

# Test info command
def test_info_command(set_test_env):
    command = "python main.py --task info"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'openai' in stdout
    assert 'anthropic' in stdout

# Test config command
def test_config_command(set_test_env):
    command = "python main.py --task config --config_key openai_api_key"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'test_openai_key' in stdout

# Test batch processing
def test_batch_processing(set_test_env):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("What is AI?\nExplain quantum computing\n")
        temp_file_path = temp_file.name

    command = f"python main.py --providers openai --task generate --batch_input {temp_file_path} --output batch_result.json"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert os.path.exists('batch_result.json')
    
    with open('batch_result.json', 'r') as f:
        results = json.load(f)
    
    assert len(results) == 2
    assert 'AI' in results[0]['result']['openai']['result'].lower()
    assert 'quantum' in results[1]['result']['openai']['result'].lower()
    
    os.remove('batch_result.json')
    os.remove(temp_file_path)

# Test streaming output
def test_streaming_output(set_test_env):
    command = "python main.py --providers openai --task generate --input 'Tell me a short story' --stream"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert len(stdout) > 0
    # Check if the output looks like it was streamed (multiple lines)
    assert stdout.count('\n') > 1

# Test multiple providers
def test_multiple_providers(set_test_env):
    command = "python main.py --providers openai anthropic --task generate --input 'What is the capital of France?'"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'openai' in stdout
    assert 'anthropic' in stdout
    assert 'Paris' in stdout

# Test error handling
def test_error_handling(set_test_env):
    command = "python main.py --providers invalid_provider --task generate --input 'Test'"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code != 0
    assert 'Error' in stderr or 'Error' in stdout

# Test interactive chat mode
@patch('builtins.input', side_effect=['Hello', 'What is AI?', 'exit'])
def test_interactive_chat(mock_input, set_test_env):
    command = "python main.py --providers openai --task chat --interactive"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'Hello' in stdout
    assert 'AI' in stdout
    assert 'exit' in stdout

# Test saving chat history
@patch('builtins.input', side_effect=['Hello', 'What is AI?', 'exit'])
def test_save_chat_history(mock_input, set_test_env):
    command = "python main.py --providers openai --task chat --interactive --save_chat"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'Chat history saved to' in stdout
    
    # Find the saved chat history file
    history_file = next(file for file in os.listdir() if file.startswith('chat_history_') and file.endswith('.json'))
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    assert len(history) > 0
    assert any('Hello' in message['content'] for message in history)
    assert any('AI' in message['content'] for message in history)
    
    os.remove(history_file)

# Test different model specification
def test_model_specification(set_test_env):
    command = "python main.py --providers openai --task generate --input 'Test' --model gpt-4o"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'gpt-4o' in stdout

# Test Azure OpenAI specific arguments
def test_azure_openai_args(set_test_env):
    command = "python main.py --providers azure_openai --task generate --input 'Test' --azure_deployment test-deployment"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'azure_openai' in stdout
    assert 'test-deployment' in stdout

# Test AWS Bedrock specific arguments
def test_aws_bedrock_args(set_test_env):
    command = "python main.py --providers aws_bedrock --task generate --input 'Test' --aws_model_id anthropic.claude-v2"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert 'aws_bedrock' in stdout
    assert 'anthropic.claude-v2' in stdout

# Test parallel processing
def test_parallel_processing(set_test_env):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("What is AI?\nExplain quantum computing\n")
        temp_file_path = temp_file.name

    command = f"python main.py --providers openai anthropic --task generate --batch_input {temp_file_path} --output parallel_result.json --parallel"
    stdout, stderr, return_code = run_cli_command(command)
    
    assert return_code == 0
    assert os.path.exists('parallel_result.json')
    
    with open('parallel_result.json', 'r') as f:
        results = json.load(f)
    
    assert len(results) == 2
    assert 'openai' in results[0]['result']
    assert 'anthropic' in results[0]['result']
    
    os.remove('parallel_result.json')
    os.remove(temp_file_path)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])