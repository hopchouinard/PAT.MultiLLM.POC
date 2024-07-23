# tests/unit/test_providers/test_aws_bedrock_provider.py

import pytest
from unittest.mock import Mock, patch
from llm_providers.aws_bedrock_provider import AWSBedrockProvider
from botocore.exceptions import ClientError
import json

@pytest.fixture
def mock_boto3():
    with patch('llm_providers.aws_bedrock_provider.boto3') as mock:
        yield mock

@pytest.fixture
def aws_bedrock_provider(mock_boto3, mock_config):
    return AWSBedrockProvider()

def test_aws_bedrock_provider_initialization(aws_bedrock_provider, mock_config, mock_boto3):
    assert isinstance(aws_bedrock_provider, AWSBedrockProvider)
    assert aws_bedrock_provider.region_name == 'us-west-2'
    assert aws_bedrock_provider.aws_access_key_id == 'test_aws_access_key_id'
    assert aws_bedrock_provider.aws_secret_access_key == 'test_aws_secret_access_key'
    mock_boto3.client.assert_called_once_with(
        service_name='bedrock-runtime',
        region_name='us-west-2',
        aws_access_key_id='test_aws_access_key_id',
        aws_secret_access_key='test_aws_secret_access_key'
    )

def test_generate(aws_bedrock_provider, mock_boto3):
    mock_response = Mock()
    mock_response['body'].read.return_value = json.dumps({"completion": "Generated text"})
    mock_boto3.client.return_value.invoke_model.return_value = mock_response

    result = aws_bedrock_provider.generate("Test prompt", model_id="anthropic.claude-v2", max_tokens=10, temperature=0.7)
    assert result == "Generated text"
    mock_boto3.client.return_value.invoke_model.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model.call_args[1]
    assert call_args['modelId'] == "anthropic.claude-v2"
    assert json.loads(call_args['body'])['prompt'] == "\n\nHuman: Test prompt\n\nAssistant:"
    assert json.loads(call_args['body'])['max_tokens_to_sample'] == 10
    assert json.loads(call_args['body'])['temperature'] == 0.7

def test_generate_with_custom_model(aws_bedrock_provider, mock_boto3):
    mock_response = Mock()
    mock_response['body'].read.return_value = json.dumps({"completion": "Custom model text"})
    mock_boto3.client.return_value.invoke_model.return_value = mock_response

    result = aws_bedrock_provider.generate("Test prompt", model_id="ai21.j2-ultra", max_tokens=20, temperature=0.5)
    assert result == "Custom model text"
    mock_boto3.client.return_value.invoke_model.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model.call_args[1]
    assert call_args['modelId'] == "ai21.j2-ultra"

def test_generate_api_error(aws_bedrock_provider, mock_boto3):
    mock_boto3.client.return_value.invoke_model.side_effect = ClientError({"Error": {"Message": "API Error"}}, "invoke_model")
    with pytest.raises(ClientError, match="API Error"):
        aws_bedrock_provider.generate("Test prompt", model_id="anthropic.claude-v2")

def test_generate_stream(aws_bedrock_provider, mock_boto3):
    mock_stream = Mock()
    mock_stream['body'].read.side_effect = [
        json.dumps({"completion": "Chunk 1"}),
        json.dumps({"completion": "Chunk 2"}),
        json.dumps({"completion": "Chunk 3"}),
        ""
    ]
    mock_boto3.client.return_value.invoke_model_with_response_stream.return_value = {'body': mock_stream}

    result = list(aws_bedrock_provider.generate_stream("Test prompt", model_id="anthropic.claude-v2", max_tokens=30, temperature=0.8))
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_boto3.client.return_value.invoke_model_with_response_stream.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model_with_response_stream.call_args[1]
    assert call_args['modelId'] == "anthropic.claude-v2"
    assert json.loads(call_args['body'])['max_tokens_to_sample'] == 30
    assert json.loads(call_args['body'])['temperature'] == 0.8

def test_generate_stream_error(aws_bedrock_provider, mock_boto3):
    mock_boto3.client.return_value.invoke_model_with_response_stream.side_effect = ClientError({"Error": {"Message": "Stream Error"}}, "invoke_model_with_response_stream")
    with pytest.raises(ClientError, match="Stream Error"):
        list(aws_bedrock_provider.generate_stream("Test prompt", model_id="anthropic.claude-v2"))

def test_embed(aws_bedrock_provider, mock_boto3):
    mock_response = Mock()
    mock_response['body'].read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]})
    mock_boto3.client.return_value.invoke_model.return_value = mock_response

    result = aws_bedrock_provider.embed("Test text", model_id="amazon.titan-embed-text-v1")
    assert result == [0.1, 0.2, 0.3]
    mock_boto3.client.return_value.invoke_model.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model.call_args[1]
    assert call_args['modelId'] == "amazon.titan-embed-text-v1"
    assert json.loads(call_args['body'])['inputText'] == "Test text"

def test_embed_error(aws_bedrock_provider, mock_boto3):
    mock_boto3.client.return_value.invoke_model.side_effect = ClientError({"Error": {"Message": "Embedding Error"}}, "invoke_model")
    with pytest.raises(ClientError, match="Embedding Error"):
        aws_bedrock_provider.embed("Test text", model_id="amazon.titan-embed-text-v1")

def test_chat(aws_bedrock_provider, mock_boto3):
    mock_response = Mock()
    mock_response['body'].read.return_value = json.dumps({"completion": "Chat response"})
    mock_boto3.client.return_value.invoke_model.return_value = mock_response

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    result = aws_bedrock_provider.chat(messages, model_id="anthropic.claude-v2", max_tokens=50, temperature=0.7)

    assert result == {
        "role": "assistant",
        "content": "Chat response",
        "model": "anthropic.claude-v2",
    }
    mock_boto3.client.return_value.invoke_model.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model.call_args[1]
    assert call_args['modelId'] == "anthropic.claude-v2"
    assert "Hello" in json.loads(call_args['body'])['prompt']
    assert "How are you?" in json.loads(call_args['body'])['prompt']

def test_chat_api_error(aws_bedrock_provider, mock_boto3):
    mock_boto3.client.return_value.invoke_model.side_effect = ClientError({"Error": {"Message": "Chat API Error"}}, "invoke_model")
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(ClientError, match="Chat API Error"):
        aws_bedrock_provider.chat(messages, model_id="anthropic.claude-v2")

def test_list_models(aws_bedrock_provider, mock_boto3):
    mock_response = {
        'modelSummaries': [
            {'modelId': 'anthropic.claude-v2', 'modelName': 'Claude V2'},
            {'modelId': 'ai21.j2-ultra', 'modelName': 'Jurassic-2 Ultra'}
        ]
    }
    mock_boto3.client.return_value.list_foundation_models.return_value = mock_response

    models = aws_bedrock_provider.list_models()
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0]['modelId'] == 'anthropic.claude-v2'
    assert models[1]['modelId'] == 'ai21.j2-ultra'

def test_get_model_info(aws_bedrock_provider, mock_boto3):
    mock_response = {
        'modelDetails': {
            'modelId': 'anthropic.claude-v2',
            'modelName': 'Claude V2',
            'providerName': 'Anthropic'
        }
    }
    mock_boto3.client.return_value.get_foundation_model.return_value = mock_response

    info = aws_bedrock_provider.get_model_info("anthropic.claude-v2")
    assert info['modelId'] == 'anthropic.claude-v2'
    assert info['modelName'] == 'Claude V2'
    assert info['providerName'] == 'Anthropic'

def test_generate_with_custom_parameters(aws_bedrock_provider, mock_boto3):
    mock_response = Mock()
    mock_response['body'].read.return_value = json.dumps({"completion": "Generated with custom parameters"})
    mock_boto3.client.return_value.invoke_model.return_value = mock_response

    result = aws_bedrock_provider.generate(
        "Test prompt",
        model_id="anthropic.claude-v2",
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        top_k=10,
        stop_sequences=["END", "STOP"]
    )
    assert result == "Generated with custom parameters"
    mock_boto3.client.return_value.invoke_model.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model.call_args[1]
    body = json.loads(call_args['body'])
    assert body['max_tokens_to_sample'] == 100
    assert body['temperature'] == 0.9
    assert body['top_p'] == 0.95
    assert body['top_k'] == 10
    assert body['stop_sequences'] == ["END", "STOP"]

def test_chat_with_system_message(aws_bedrock_provider, mock_boto3):
    mock_response = Mock()
    mock_response['body'].read.return_value = json.dumps({"completion": "Chat response with system message"})
    mock_boto3.client.return_value.invoke_model.return_value = mock_response

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    result = aws_bedrock_provider.chat(messages, model_id="anthropic.claude-v2")
    assert result["content"] == "Chat response with system message"
    mock_boto3.client.return_value.invoke_model.assert_called_once()
    call_args = mock_boto3.client.return_value.invoke_model.call_args[1]
    prompt = json.loads(call_args['body'])['prompt']
    assert "You are a helpful assistant." in prompt
    assert "Hello, who are you?" in prompt

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])