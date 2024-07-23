# LLM API Manager Documentation

## Table of Contents

1. [User Guide](#user-guide)
   1. [Installation](#installation)
   2. [Configuration](#configuration)
   3. [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
   4. [API Usage](#api-usage)
2. [Technical Guide](#technical-guide)
   1. [Project Structure](#project-structure)
   2. [Adding a New Provider](#adding-a-new-provider)
   3. [Testing](#testing)
   4. [Extending Functionality](#extending-functionality)

## User Guide

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/hopchouinard/PAT.MultiLLM.POC.git
   cd llm-api-manager
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the project root directory.
2. Add your API keys and other configuration settings:

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

### Command-Line Interface (CLI) Usage

The main script `main.py` provides a command-line interface for interacting with the LLM API Manager.

#### General Command Structure

```
python main.py --providers <provider1> <provider2> --task <task> [additional options]
```

#### Available Tasks

- `generate`: Generate text
- `embed`: Create embeddings
- `chat`: Engage in a chat conversation
- `info`: Get information about providers
- `config`: View or modify configuration

#### Common Options

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

#### Examples

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

### API Usage

The LLM API Manager can be used as a Python library in your projects.

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
        print(f"{provider} response: {result['result']}")
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
        print(f"{provider} embedding dimension: {len(result['result'])}")
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
        print(f"{provider} response: {result['result']['content']}")
    else:
        print(f"{provider} error: {result['error']}")
```

#### Parallel Processing

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

#### Getting Provider Information

```python
info = manager.get_provider_info()
print(json.dumps(info, indent=2))
```

## Technical Guide

### Project Structure

The LLM API Manager project has the following structure:

```
llm-api-manager/
├── llm_providers/
│   ├── __init__.py
│   ├── base.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   ├── ollama_provider.py
│   ├── lmstudio_provider.py
│   ├── azure_openai_provider.py
│   └── aws_bedrock_provider.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   └── logging.py
├── tests/
│   ├── unit/
│   │   ├── test_providers/
│   │   │   ├── test_openai_provider.py
│   │   │   ├── test_anthropic_provider.py
│   │   │   └── ...
│   │   └── test_utils/
│   │       ├── test_config.py
│   │       └── test_logging.py
│   ├── integration/
│   │   ├── test_llm_manager_integration.py
│   │   └── test_provider_integration.py
│   └── end_to_end/
│       └── test_cli.py
├── llm_manager.py
├── main.py
├── requirements.txt
└── README.md
```

### Adding a New Provider

To add support for a new LLM provider:

1. Create a new file in the `llm_providers` directory (e.g., `new_provider.py`).
2. Implement a class that inherits from `LLMProvider` and implements all required methods:

```python
from .base import LLMProvider

class NewProvider(LLMProvider):
    def __init__(self):
        # Initialize the provider with necessary configurations

    def generate(self, prompt: str, **kwargs) -> str:
        # Implement text generation

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        # Implement streaming text generation

    def embed(self, text: str, **kwargs) -> List[float]:
        # Implement text embedding

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        # Implement chat functionality

    def list_available_models(self) -> List[Dict[str, Any]]:
        # Implement method to list available models
```

3. Add the new provider to the `PROVIDER_MAP` in `llm_providers/__init__.py`:

```python
from .new_provider import NewProvider

PROVIDER_MAP: Dict[str, Type[LLMProvider]] = {
    # ... existing providers ...
    "new_provider": NewProvider,
}
```

4. Update the `Config` class in `utils/config.py` to include any new configuration options for the new provider.

5. Add unit tests for the new provider in `tests/unit/test_providers/test_new_provider.py`.

6. Update integration tests in `tests/integration/test_provider_integration.py` to include the new provider.

7. Update the CLI in `main.py` to handle any specific arguments or options for the new provider.

### Testing

The project uses pytest for testing. The test suite is organized into unit tests, integration tests, and end-to-end tests.

To run the tests:

1. Install the testing dependencies:
   ```
   pip install pytest pytest-asyncio
   ```

2. Run all tests:
   ```
   pytest
   ```

3. Run specific test categories:
   ```
   pytest tests/unit  # Run unit tests
   pytest tests/integration  # Run integration tests
   pytest tests/end_to_end  # Run end-to-end tests
   ```

4. Run tests with coverage:
   ```
   pytest --cov=llm_providers --cov=utils --cov=llm_manager
   ```

When adding new functionality or providers, make sure to update or add corresponding tests.

### Extending Functionality

To extend the functionality of the LLM API Manager:

1. Identify the component that needs to be extended (e.g., `LLMManager`, specific provider, CLI interface).

2. Implement the new functionality in the appropriate file(s).

3. If adding new methods to `LLMManager`, update the `LLMProvider` base class in `llm_providers/base.py` if necessary.

4. Update all provider classes to implement any new methods added to the base class.

5. Add appropriate error handling and logging for the new functionality.

6. Update the CLI interface in `main.py` to expose the new functionality if applicable.

7. Add unit tests for the new functionality in the appropriate test files.

8. Update integration tests to cover the new functionality.

9. Update this documentation to reflect the new features and usage.

10. If the new functionality requires additional dependencies, add them to `requirements.txt`.

Example of adding a new method to `LLMManager`:

```python
class LLMManager:
    # ... existing methods ...

    def new_functionality(self, input_data: str, **kwargs) -> Dict[str, Any]:
        results = {}
        for provider_name, provider in self.providers.items():
            try:
                result = provider.new_functionality(input_data, **kwargs)
                results[provider_name] = {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error in {provider_name} new_functionality: {str(e)}", exc_info=True)
                results[provider_name] = {"status": "error", "error": str(e)}
        return results
```

Then, update each provider class to implement the `new_functionality` method, and add corresponding tests.

By following these guidelines, you can maintain the project's structure and consistency while extending its capabilities.
