# PAT.MultiLLM.POC
# LLM API Manager

LLM API Manager is a powerful and flexible command-line tool for interacting with multiple Language Model (LLM) providers. It provides a unified interface for text generation, embeddings, and chat functionalities across various LLM APIs.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic, Google's Gemini, Ollama, LM Studio)
- Text generation with customizable parameters
- Embedding generation
- Interactive chat mode with conversation history
- Batch processing of inputs with parallel execution
- Streaming output for real-time text generation
- Progress bar for batch processing tasks
- Error handling and automatic retries for failed API calls
- Configuration management via command-line
- Logging with customizable log levels

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hopchouinard/PAT.MULTILLM.POC.git
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
   OLLAMA_HOST = http://localhost:11434
   LMSTUDIO_API_BASE = http://localhost:1234/v1
   ```

## Usage

The main script `main.py` provides a command-line interface for interacting with the LLM API Manager. Here are some example use cases:

### Text Generation

Generate text using a specific provider:
```
python main.py --providers openai --task generate --input "Explain quantum computing" --max_tokens 150
```

### Embedding Generation

Create embeddings using multiple providers:
```
python main.py --providers openai anthropic --task embed --input "Quantum computing is fascinating"
```

### Interactive Chat

Start an interactive chat session:
```
python main.py --providers anthropic --task chat --interactive --save_chat
```

### Batch Processing

Process multiple inputs in parallel:
```
python main.py --providers openai anthropic --task generate --batch_input inputs.txt --output results.json --parallel
```

### Configuration Management

View or modify configuration settings:
```
python main.py --task config --config_key openai_api_key
python main.py --task config --config_key openai_api_key --config_value new_key_here
```

## Project Structure

- `main.py`: The main script that provides the command-line interface.
- `llm_manager.py`: Contains the `LLMManager` class that orchestrates interactions with different LLM providers.
- `llm_providers/`: Directory containing individual provider implementations.
- `utils/`: Directory containing utility modules for configuration, logging, etc.

## Contributing

Contributions to the LLM API Manager are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and include tests for new functionalities.

# LLM API Manager

## API Documentation

The LLM API Manager can be used as a module in your Python projects. Here's how to use its main components programmatically:

### LLMManager Class

The `LLMManager` class is the main interface for interacting with different LLM providers.

#### Initialization

```python
from llm_manager import LLMManager

# Initialize with specific providers
manager = LLMManager(providers=["openai", "anthropic"])

# Initialize with all available providers
manager = LLMManager()
```

#### Text Generation

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
        print(f"{provider} response: {result['response']}")
    else:
        print(f"{provider} error: {result['error']}")
```

#### Streaming Text Generation

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

#### Embedding Generation

```python
results = manager.embed(
    text="Quantum computing is fascinating",
    provider="openai",  # Optional: specify a provider
    model="text-embedding-ada-002"  # Optional: specify a model
)

for provider, result in results.items():
    if result['status'] == 'success':
        print(f"{provider} embedding dimension: {len(result['embedding'])}")
    else:
        print(f"{provider} error: {result['error']}")
```

#### Chat Functionality

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
        print(f"{provider} response: {result['response']['content']}")
    else:
        print(f"{provider} error: {result['error']}")
```

#### Parallel Processing

```python
inputs = ["Explain quantum computing", "Describe machine learning", "What is artificial intelligence?"]

results = manager.generate_parallel(
    prompts=inputs,
    max_tokens=100,
    temperature=0.7
)

for input_text, result in zip(inputs, results):
    print(f"Input: {input_text}")
    for provider, response in result.items():
        if response['status'] == 'success':
            print(f"  {provider} response: {response['response']}")
        else:
            print(f"  {provider} error: {response['error']}")
    print("-" * 40)
```

### Provider-Specific Classes

Each LLM provider has its own class that inherits from the `LLMProvider` base class. These can be used directly if you need provider-specific functionality:

```python
from llm_providers import OpenAIProvider, AnthropicProvider

openai_provider = OpenAIProvider()
anthropic_provider = AnthropicProvider()

openai_response = openai_provider.generate("Explain quantum computing")
anthropic_response = anthropic_provider.generate("Explain quantum computing")
```

### Utility Functions

The `utils` module provides several utility functions:

```python
from utils import get_config, set_log_level, logger

# Get configuration
config = get_config()
api_key = config.get('openai_api_key')

# Set log level
set_log_level('DEBUG')

# Use logger
logger.info("This is an info message")
logger.error("This is an error message")
```

## Error Handling

The LLM API Manager uses exception handling to manage errors. When using the `LLMManager` class, errors are typically returned in the result dictionary with a 'status' of 'error'. When using provider classes directly, exceptions may be raised and should be caught and handled in your code.

```python
try:
    result = openai_provider.generate("Explain quantum computing")
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

For more detailed information about each class and method, please refer to the inline documentation in the source code.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This project makes use of the following open-source libraries:
- OpenAI Python Client
- Anthropic AI API
- Google Generative AI
- Ollama
- tqdm

We thank the developers of these libraries for their contributions to the open-source community.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact the project maintainer at [your-email@example.com].