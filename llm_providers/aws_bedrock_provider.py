import boto3
import json
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Generator
from .base import LLMProvider
from utils.config import get_config
from utils.logging import logger

class AWSBedrockProvider(LLMProvider):
    """
    AWS Bedrock API provider for language model operations.
    """

    def __init__(self):
        """
        Initialize the AWS Bedrock provider with API details from configuration.
        """
        config = get_config()
        self.region_name = config.get('aws_region_name')
        self.aws_access_key_id = config.get('aws_access_key_id')
        self.aws_secret_access_key = config.get('aws_secret_access_key')
        
        if not self.region_name or not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.error("AWS credentials or region not found in configuration")
            raise ValueError("AWS credentials or region not found in configuration")
        
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )
        
        logger.info("AWSBedrockProvider initialized successfully")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional arguments for the API call.

        Returns:
            str: The generated text.

        Raises:
            Exception: If there's an error in the API call.
        """
        model_id = kwargs.get('model_id', 'anthropic.claude-v2')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text with model: {model_id}")
        try:
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": 1,
            })

            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body.get('completion', '').strip()
        except ClientError as e:
            logger.error(f"AWS Bedrock API error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in AWS Bedrock API call: {str(e)}", exc_info=True)
            raise

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate text in a streaming fashion based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional arguments for the API call.

        Yields:
            str: Chunks of generated text.

        Raises:
            Exception: If there's an error in the API call.
        """
        model_id = kwargs.get('model_id', 'anthropic.claude-v2')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Generating text stream with model: {model_id}")
        try:
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": 1,
            })

            response = self.bedrock_client.invoke_model_with_response_stream(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )

            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'])
                if 'completion' in chunk:
                    yield chunk['completion']
        except ClientError as e:
            logger.error(f"AWS Bedrock API streaming error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in AWS Bedrock API streaming call: {str(e)}", exc_info=True)
            raise

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate embeddings for the given text.

        Args:
            text (str): The input text to embed.
            **kwargs: Additional arguments for the API call.

        Returns:
            List[float]: The embedding vector.

        Raises:
            Exception: If there's an error in the API call.
        """
        model_id = kwargs.get('model_id', 'amazon.titan-embed-text-v1')
        
        logger.info(f"Generating embedding with model: {model_id}")
        try:
            body = json.dumps({
                "inputText": text
            })

            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body.get('embedding', [])
        except ClientError as e:
            logger.error(f"AWS Bedrock API embedding error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in AWS Bedrock embedding: {str(e)}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Engage in a chat conversation with the AWS Bedrock model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries. Each dictionary
                should have 'role' (either 'user' or 'assistant') and 'content' keys.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict[str, Any]: The model's response containing the message and other metadata.

        Raises:
            Exception: If there's an error in the API call.
        """
        model_id = kwargs.get('model_id', 'anthropic.claude-v2')
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        logger.info(f"Starting chat with model: {model_id}")
        try:
            # Format messages for Claude model
            formatted_messages = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
            formatted_messages += "\n\nAssistant:"

            body = json.dumps({
                "prompt": formatted_messages,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": 1,
            })

            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return {
                "role": "assistant",
                "content": response_body.get('completion', '').strip(),
                "model": model_id,
            }
        except ClientError as e:
            logger.error(f"AWS Bedrock API chat error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in AWS Bedrock API chat call: {str(e)}", exc_info=True)
            raise

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from AWS Bedrock.

        Returns:
            List[Dict[str, Any]]: A list of available models and their details.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info("Fetching available models from AWS Bedrock")
        try:
            bedrock = boto3.client(
                service_name='bedrock',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            response = bedrock.list_foundation_models()
            return response.get('modelSummaries', [])
        except ClientError as e:
            logger.error(f"Error fetching AWS Bedrock models: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching AWS Bedrock models: {str(e)}", exc_info=True)
            raise

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id (str): The ID of the model to retrieve information for.

        Returns:
            Dict[str, Any]: Information about the model.

        Raises:
            Exception: If there's an error in the API call.
        """
        logger.info(f"Getting model info for: {model_id}")
        try:
            bedrock = boto3.client(
                service_name='bedrock',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            response = bedrock.get_foundation_model(modelIdentifier=model_id)
            return response.get('modelDetails', {})
        except ClientError as e:
            logger.error(f"Error retrieving AWS Bedrock model info: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving AWS Bedrock model info: {str(e)}", exc_info=True)
            raise