# main.py

import argparse
import json
import sys
from typing import List, Dict
from llm_manager import LLMManager
from utils import logger, set_log_level, get_config, Config
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM API Manager")
    parser.add_argument("--providers", nargs="+", help="List of providers to use")
    parser.add_argument("--task", choices=["generate", "embed", "chat", "info", "config"], required=True, help="Task to perform")
    parser.add_argument("--input", help="Input text for generation or embedding")
    parser.add_argument("--chat_history", help="JSON file containing chat history")
    parser.add_argument("--output", help="Output file to save results")
    parser.add_argument("--model", help="Specific model to use (if applicable)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Set the logging level")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for chat")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output for generation")
    parser.add_argument("--batch_input", help="File containing multiple inputs for batch processing")
    parser.add_argument("--config_key", help="Configuration key to view or modify")
    parser.add_argument("--config_value", help="New value for the configuration key")
    parser.add_argument("--save_chat", action="store_true", help="Save chat history in interactive mode")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing for batch inputs")
    return parser.parse_args()

def load_chat_history(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_results(results: Dict, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def interactive_chat(manager: LLMManager, initial_history: List[Dict[str, str]], args):
    history = initial_history
    print("Enter your messages. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        history.append({"role": "user", "content": user_input})
        results = manager.chat(history, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
        for provider, result in results.items():
            if result['status'] == 'success':
                response = result['response']['content']
                print(f"{provider}: {response}")
                history.append({"role": "assistant", "content": response})
            else:
                print(f"{provider} error: {result['error']}")
        
        if args.save_chat:
            save_chat_history(history)
    
    return history

def save_chat_history(history: List[Dict[str, str]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Chat history saved to {filename}")

def stream_generate(manager: LLMManager, prompt: str, args):
    for chunk in manager.generate_stream(prompt, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature):
        if chunk['status'] == 'success':
            print(chunk['chunk'], end='', flush=True)
        else:
            print(f"\nError: {chunk['error']}")
    print()  # New line after streaming is complete

def batch_process(manager: LLMManager, input_file: str, args):
    with open(input_file, 'r') as f:
        inputs = f.readlines()
    
    if args.parallel:
        return parallel_batch_process(manager, inputs, args)
    else:
        return sequential_batch_process(manager, inputs, args)

def sequential_batch_process(manager: LLMManager, inputs: List[str], args):
    results = []
    for input_text in tqdm(inputs, desc="Processing inputs"):
        input_text = input_text.strip()
        if args.task == 'generate':
            result = manager.generate(input_text, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
        elif args.task == 'embed':
            result = manager.embed(input_text, model=args.model)
        else:
            logger.error(f"Batch processing not supported for task: {args.task}")
            return
        results.append({"input": input_text, "result": result})
    return results

def parallel_batch_process(manager: LLMManager, inputs: List[str], args):
    def process_input_with_retry(input_text):
        input_text = input_text.strip()
        for attempt in range(MAX_RETRIES):
            try:
                if args.task == 'generate':
                    return {"input": input_text, "result": manager.generate(input_text, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)}
                elif args.task == 'embed':
                    return {"input": input_text, "result": manager.embed(input_text, model=args.model)}
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for input: {input_text}. Retrying...")
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for input: {input_text}. Error: {str(e)}")
                    return {"input": input_text, "error": str(e)}
    
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_input = {executor.submit(process_input_with_retry, input_text): input_text for input_text in inputs}
        for future in tqdm(as_completed(future_to_input), total=len(inputs), desc="Processing inputs"):
            result = future.result()
            results.append(result)
    
    return results

def handle_config(config: Config, args):
    if args.config_key:
        if args.config_value:
            config.set(args.config_key, args.config_value)
            print(f"Updated {args.config_key} to {args.config_value}")
        else:
            value = config.get(args.config_key)
            print(f"{args.config_key}: {value}")
    else:
        print(json.dumps(config.get_all(), indent=2))

def main():
    args = parse_arguments()
    set_log_level(args.log_level)
    
    config = get_config()
    logger.info(f"Using configuration: {config.get_all()}")

    if args.task == "config":
        handle_config(config, args)
        return

    manager = LLMManager(args.providers)

    if args.batch_input:
        results = batch_process(manager, args.batch_input, args)
        if results:
            print(json.dumps(results, indent=2))
        if args.output:
            save_results(results, args.output)
        return

    if args.task == "generate":
        if not args.input:
            logger.error("Input text is required for generation task")
            return
        if args.stream:
            stream_generate(manager, args.input, args)
        else:
            results = manager.generate(args.input, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
            for provider, result in results.items():
                if result['status'] == 'success':
                    print(f"{provider} response:")
                    print(result['response'])
                    print("-" * 40)
                else:
                    print(f"{provider} error: {result['error']}")

    elif args.task == "embed":
        if not args.input:
            logger.error("Input text is required for embedding task")
            return
        results = manager.embed(args.input, model=args.model)
        for provider, result in results.items():
            if result['status'] == 'success':
                print(f"{provider} embedding dimension: {len(result['embedding'])}")
            else:
                print(f"{provider} error: {result['error']}")

    elif args.task == "chat":
        if args.chat_history:
            chat_history = load_chat_history(args.chat_history)
        else:
            chat_history = []
        
        if args.interactive:
            chat_history = interactive_chat(manager, chat_history, args)
        else:
            if args.input:
                chat_history.append({"role": "user", "content": args.input})
            
            results = manager.chat(chat_history, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
            for provider, result in results.items():
                if result['status'] == 'success':
                    print(f"{provider} response:")
                    print(result['response']['content'])
                    print("-" * 40)
                else:
                    print(f"{provider} error: {result['error']}")

    elif args.task == "info":
        info = manager.get_provider_info()
        print(json.dumps(info, indent=2))

    if args.output:
        save_results(results, args.output)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()