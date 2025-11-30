"""Persistence - saving/loading processed conversations with metadata."""
import json
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


def get_git_commit() -> str:
    """Gets the current git commit hash or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def create_metadata(
    model: str,
    llm_mode: str,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Creates metadata for the processed conversations."""
    from .config import SUMMARIZATION_PROMPT, MEMORY_EXTRACTION_PROMPT

    metadata = {
        "processing_timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "model": model,
        "llm_mode": llm_mode,
        "prompts": {
            "summarization": SUMMARIZATION_PROMPT.strip(),
            "memory_extraction": MEMORY_EXTRACTION_PROMPT.strip()
        }
    }

    if seed is not None:
        metadata["random_seed"] = seed

    if n_samples is not None:
        metadata["n_samples"] = n_samples

    return metadata


def save_conversations(
    conversations: List[Dict[str, Any]],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Saves conversations to a JSON file with pretty formatting and optional metadata."""
    output = {
        "conversations": conversations
    }

    if metadata:
        output["metadata"] = metadata

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"Conversation history successfully saved to {filename}")
        if metadata:
            print(f"  Model: {metadata.get('model')}")
            print(f"  LLM Mode: {metadata.get('llm_mode')}")
            print(f"  Git Commit: {metadata.get('git_commit')}")
    except IOError as e:
        print(f"Error saving file {filename}: {e}")


def load_processed_conversations(filename: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Loads processed conversations from a JSON file, returns (conversations, metadata)."""
    if not os.path.exists(filename):
        print(f"File {filename} not found. Starting with an empty conversation history.")
        return [], None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Handle both old format (direct list) and new format (with metadata wrapper)
            if isinstance(data, list):
                print(f"Conversation history successfully loaded from {filename} (legacy format)")
                return data, None
            elif isinstance(data, dict):
                conversations = data.get('conversations', [])
                metadata = data.get('metadata')
                print(f"Conversation history successfully loaded from {filename}")
                if metadata:
                    print(f"  Processed with: {metadata.get('model')} ({metadata.get('llm_mode')})")
                    print(f"  Git commit: {metadata.get('git_commit')}")
                return conversations, metadata
            else:
                print(f"Unexpected data format in {filename}")
                return [], None

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}. Returning an empty list.")
        return [], None
    except IOError as e:
        print(f"Error reading file {filename}: {e}. Returning an empty list.")
        return [], None


def generate_output_filename(model_name: str, n_samples: Optional[int], is_prod: bool) -> str:
    """Generates output filename based on date, model, and sample count."""
    # Get date in MMDD format
    date_str = datetime.now().strftime('%m%d')

    # Clean model name (remove special chars)
    clean_model = model_name.replace(':', '').replace('.', '').replace('-', '')

    # Build filename parts
    parts = [f"convs_with_memories_{date_str}_{clean_model}"]

    if n_samples:
        parts.append(f"nsamples{n_samples}")

    # Choose directory
    directory = "data/prod" if is_prod else "data/debug"

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    filename = "_".join(parts) + ".json"
    return os.path.join(directory, filename)
