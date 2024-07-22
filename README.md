# LLM API Manager

LLM API Manager is a powerful and flexible tool for interacting with multiple Language Model (LLM) providers. It provides a unified interface for text generation, embeddings, and chat functionalities across various LLM APIs, both through a command-line interface and as a Python library.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic, Google's Gemini, Ollama, LM Studio, Azure OpenAI, AWS Bedrock)
- Text generation with customizable parameters
- Embedding generation
- Interactive chat mode with conversation history
- Batch processing of inputs with parallel execution
- Streaming output for real-time text generation
- Progress bar for batch processing tasks
- Error handling and automatic retries for failed API calls
- Configuration management via command-line and programmatic interface
- Logging with customizable log levels

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hopchouinard/PAT.MultiLLM.POC.git
   cd llm-api-manager
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GEMINI_API_KEY=your_gemini_api_key
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_API_BASE=your_azure_openai_endpoint
   AWS_ACCESS_KEY_ID=your_aws_access_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
   AWS_REGION_NAME=your_aws_region
   OLLAMA_HOST=http://localhost:11434
   LMSTUDIO_API_BASE=http://localhost:1234/v1
   ```

## Command-Line Interface (CLI) Usage

The main script `main.py` provides a command-line interface for interacting with the LLM API Manager.

### General Command Structure

```
python main.py --providers <provider1> <provider2> --task <task> [additional options]
```

### Available Tasks

- `generate`: Generate text
- `embed`: Create embeddings
- `chat`: Engage in a chat conversation
- `info`: Get information about providers
- `config`: View or modify configuration

### Common Options

- `--input`: Input text for generation or embedding
- `--output`: Output file to save results
- `--model`: Specific model to use (if applicable)
- `--max_tokens`: Maximum number of tokens to generate
- `--temperature`: Temperature for text generation
- `--log_level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--interactive`: Enable interactive mode for chat
- `--stream`: Enable streaming output for generation
- `--batch_input`: File containing multiple inputs for batch processing
- `--parallel`: Enable parallel processing for batch inputs

### Examples

1. Generate text using a specific provider:
   ```
   python main.py --providers openai --task generate --input "Explain quantum computing" --max_tokens 150
   ```

2. Create embeddings using multiple providers:
   ```
   python main.py --providers openai anthropic --task embed --input "Quantum computing is fascinating"
   ```

3. Start an interactive chat session:
   ```
   python main.py --providers anthropic --task chat --interactive
   ```

4. Process multiple inputs in parallel:
   ```
   python main.py --providers openai anthropic --task generate --batch_input inputs.txt --output results.json --parallel
   ```

5. View or modify configuration:
   ```
   python main.py --task config --config_key openai_api_key
   python main.py --task config --config_key openai_api_key --config_value new_key_here
   ```

## Programmatic Usage

The LLM API Manager can also be used as a Python library in your projects.

### Initialization

```python
from llm_manager import LLMManager

# Initialize with specific providers
manager = LLMManager(providers=["openai", "anthropic"])

# Initialize with all available providers
manager = LLMManager()
```

### Text Generation

```python
results = manager.generate(
    prompt="Explain quantum computing",
    provider="openai",  # Optional: specify a provider
    model="text-davinci-002",  # Optional: specify a model
    max_tokens=150,
    temperature=0.7
)

for provider, result in results.items():
    if result['status'] == 'success':
        print(f"{provider} response: {result['result']}")
    else:
        print(f"{provider} error: {result['error']}")
```

### Streaming Text Generation

```python
for chunk in manager.generate_stream(
    prompt="Tell me a story",
    provider="anthropic",
    model="claude-2",
    max_tokens=200
):
    if chunk['status'] == 'success':
        print(chunk['chunk'], end='', flush=True)
    else:
        print(f"\nError: {chunk['error']}")
```

### Embedding Generation

```python
results = manager.embed(
    text="Quantum computing is fascinating",
    provider="openai",  # Optional: specify a provider
    model="text-embedding-ada-002"  # Optional: specify a model
)

for provider, result in results.items():
    if result['status'] == 'success':
        print(f"{provider} embedding dimension: {len(result['result'])}")
    else:
        print(f"{provider} error: {result['error']}")
```

### Chat Functionality

```python
chat_history = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's a famous landmark there?"}
]

results = manager.chat(
    messages=chat_history,
    provider="openai",  # Optional: specify a provider
    model="gpt-3.5-turbo",  # Optional: specify a model
    max_tokens=50
)

for provider, result in results.items():
    if result['status'] == 'success':
        print(f"{provider} response: {result['result']['content']}")
    else:
        print(f"{provider} error: {result['error']}")
```

### Parallel Processing

```python
prompts = ["Explain quantum computing", "Describe machine learning", "What is artificial intelligence?"]

results = manager.generate_parallel(prompt=prompts, max_tokens=100, temperature=0.7)

for prompt, result in zip(prompts, results.values()):
    print(f"Prompt: {prompt}")
    for provider, response in result.items():
        if response['status'] == 'success':
            print(f"  {provider} response: {response['result']}")
        else:
            print(f"  {provider} error: {response['error']}")
    print("-" * 40)
```

### Getting Provider Information

```python
info = manager.get_provider_info()
print(json.dumps(info, indent=2))
```

## Error Handling

The LLM API Manager uses a standardized approach to error handling. All methods return a dictionary with a 'status' key, which is either 'success' or 'error'. In case of an error, an 'error' key is included with the error message.

## Extending the LLM API Manager

To add support for a new LLM provider:

1. Create a new file in the `llm_providers` directory (e.g., `new_provider.py`).
2. Implement a class that inherits from `LLMProvider` and implements all required methods.
3. Add the new provider to the `PROVIDER_MAP` in `llm_providers/__init__.py`.

## Contributing

Contributions to the LLM API Manager are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and include tests for new functionalities.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This project makes use of the following open-source libraries:
- OpenAI Python Client
- Anthropic AI API
- Google Generative AI
- Ollama
- Boto3 (AWS SDK)
- Azure OpenAI SDK
- tqdm

We thank the developers of these libraries for their contributions to the open-source community.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact the project maintainer at [52129156+hopchouinard@users.noreply.github.com].