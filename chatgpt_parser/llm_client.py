"""LLM client initialization and detection (OpenAI and Ollama support)."""
import os
from typing import Tuple
from openai import OpenAI


def setup_llm_client(model_name: str) -> Tuple[OpenAI, str, str]:
    """Sets up LLM client based on model name, returns (client, model_name, llm_mode)."""
    # Detect if it's a local/ollama model (typically contains ":")
    if ":" in model_name or model_name.startswith("llama") or model_name.startswith("mistral"):
        # Ollama client with OpenAI-compatible API
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # required but unused
        )
        llm_mode = "local"
        print(f"Using local Ollama model: {model_name}")
    else:
        # Assume OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        llm_mode = "openai"
        print(f"Using OpenAI model: {model_name}")

    return client, model_name, llm_mode
