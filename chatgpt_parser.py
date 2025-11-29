"""ChatGPT conversation parser and processor with LLM-based summarization and memory extraction."""
import json
import os
import sys
import argparse
import subprocess
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI


# Prompts for LLM operations
SUMMARIZATION_PROMPT = """
You are an expert in summarizing content. You will be given a conversation between an agent and a human and your task is to summarise the last message content in a way that no content
is lost but we avoid all the details that are not relevant to understand the conversation the user is having. give brief simple bullet points (no nested, no formatting) that cover the topic of what was talked.
The content should be the minimum for someone reading the agent's summary to understand the users's response.
Keep the language of the conversation.

[OUTPUT EXAMPLE]
- Explica al usuario qué evitar, cómo aliviar el dolor, qué alimentos consumir suaves.
- Mejora típica en 48-72 horas, sugiere consultar con un médico si persisten síntomas graves.
[END OF EXAMPLE]

This is the turn to summarise:
"""

MEMORY_EXTRACTION_PROMPT = """
[System Role] You are a dedicated Memory Manager. Your sole purpose is to extract actionable, long-term user data from conversations to build a personalized user profile.

[Extraction Criteria] Extract details ONLY if they fall into these categories:
- Explicit Preferences: Likes, dislikes, dietary restrictions, favorite items/media.
- Biographical current or historic facts: Name, location, job, age, family members, pets, facts about previous life.
- Recurring Routines: Daily habits, schedules, frequent activities.
- Future Intent: Specific upcoming plans, goals, or milestones.

[Exclusion Criteria]
- Ignore temporary states (e.g., "I am hungry now").
- Ignore general conversation topics or opinions unless they indicate a strong preference.
- Ignore summaries of the chat.

[Format Constraints]
- Output strictly a bulleted list.
- Do not include introductory or concluding text.
- The facts that you store should contain all the relevant context that make that piece of data wholesome.
- If no relevant data is found, output "None".

[Input Conversation]
"""


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


def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """Loads the raw conversations.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []


def extract_linear_conversation(conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Traverses the conversation graph backwards from 'current_node' to reconstruct linear chat history."""
    mapping = conversation_data.get('mapping', {})
    current_node_id = conversation_data.get('current_node')

    messages = []

    # Traverse backwards from the leaf (latest message) to the root
    while current_node_id:
        node = mapping.get(current_node_id)
        if not node:
            break

        message_data = node.get('message')

        # Some nodes are metadata/system updates with no message content
        if message_data and message_data.get('content'):
            content = message_data['content']
            parts = content.get('parts', [])

            # Filter out non-text parts (like empty lists or plugin calls)
            text_content = ""
            if isinstance(parts, list) and len(parts) > 0:
                # Join parts if they are strings (handles standard text)
                text_content = "".join([str(p) for p in parts if isinstance(p, str)])

            if text_content:
                timestamp = message_data.get('create_time')
                readable_time = None
                if timestamp:
                    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                clean_msg = {
                    "role": message_data['author']['role'],
                    "text": text_content,
                    "timestamp": readable_time,
                    "model_slug": message_data.get('metadata', {}).get('model_slug')
                }
                messages.append(clean_msg)

        # Move pointer to the parent
        current_node_id = node.get('parent')

    # The loop extracted them in reverse (Newest -> Oldest), so flip it back
    return messages[::-1]


def filter_conversations(
    all_conversations: List[Dict[str, Any]],
    keyword: Optional[str] = None,
    min_turns: int = 0
) -> List[Dict[str, Any]]:
    """Filters conversations by keyword or length, returns list of cleaned linear objects."""
    cleaned_data = []

    for conv in all_conversations:
        title = conv.get('title', 'Untitled')

        # Basic filtering logic
        if keyword and keyword.lower() not in title.lower():
            continue

        linear_chat = extract_linear_conversation(conv)

        # Filter out short "hello/goodbye" chats
        if len(linear_chat) < min_turns:
            continue

        cleaned_data.append({
            "id": conv.get('id'),
            "title": title,
            "messages": linear_chat
        })

    return cleaned_data


def flatten_turn(turn: Dict[str, str], agent_mode: str = 'summarised_conversation') -> str:
    """Flattens a turn dictionary to a formatted string for LLM input."""
    for key, text in turn.items():
        if key == 'user':
            return f"[user]\n{text}\n"
        if key == agent_mode:
            return f"[agent]\n{text}\n"
    return ''


def summarize_assistant_turn(
    client: OpenAI,
    model: str,
    turn_text: str,
    conversation_context: str
) -> str:
    """Summarizes an assistant's turn using the LLM."""
    prompt = SUMMARIZATION_PROMPT + conversation_context + f"assistant: {turn_text}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def process_conversation_with_summaries(
    client: OpenAI,
    model: str,
    conversation: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """Processes a conversation and adds summaries to all assistant turns."""
    processed = {
        "id": conversation['id'],
        "title": conversation['title'],
        "create_time": conversation['create_time'],
        "update_time": conversation['update_time'],
        "turns": []
    }

    linear_messages = extract_linear_conversation(conversation)

    for message in linear_messages:
        if message['role'] == 'user':
            turn = {message['role']: message['text']}
            if verbose:
                print(f"[USER] {message['text'][:50]}...")
        elif message['role'] == 'assistant':
            # Build context from previous turns
            context = "\n".join([flatten_turn(t) for t in processed['turns']])

            # Summarize the assistant turn
            summary = summarize_assistant_turn(
                client,
                model,
                message['text'],
                context
            )

            turn = {
                message['role']: message['text'],
                "summarised_conversation": summary
            }

            if verbose:
                print(f"[ASSISTANT SUMMARY] {summary[:50]}...")
        else:
            # Skip system messages
            continue

        processed['turns'].append(turn)

    return processed


def extract_memories(
    client: OpenAI,
    model: str,
    conversation: Dict[str, Any]
) -> str:
    """Extracts long-term memories from a conversation."""
    # Flatten all turns for memory extraction
    conversation_text = "\n".join([
        flatten_turn(turn) for turn in conversation.get('turns', [])
    ])

    prompt = MEMORY_EXTRACTION_PROMPT + conversation_text

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


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


def create_metadata(model: str, llm_mode: str, seed: Optional[int] = None, n_samples: Optional[int] = None) -> Dict[str, Any]:
    """Creates metadata for the processed conversations."""
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Process ChatGPT conversations with LLM-based summarization and memory extraction'
    )

    parser.add_argument(
        '--llm',
        type=str,
        required=True,
        help='Model name to use (e.g., "llama3.1:8b", "gpt-4o-mini")'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/conversations.json',
        help='Input conversations.json file path (default: data/conversations.json)'
    )

    parser.add_argument(
        '--sample',
        type=int,
        help='Number of conversations to sample (default: process all)'
    )

    parser.add_argument(
        '--prod',
        action='store_true',
        help='Save to data/prod/ instead of data/debug/'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--no-memories',
        action='store_true',
        help='Skip memory extraction (only generate summaries)'
    )

    args = parser.parse_args()

    # Setup LLM client
    print(f"Setting up client for model: {args.llm}...")
    client, model, llm_mode = setup_llm_client(model_name=args.llm)

    # Generate output filename
    output_file = generate_output_filename(model, args.sample, args.prod)
    print(f"Output will be saved to: {output_file}")

    # Load conversations
    print(f"\nLoading conversations from {args.input}...")
    raw_data = load_conversations(args.input)

    if not raw_data:
        print("No conversations found. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(raw_data)} conversations")

    # Set random seed based on mode
    seed = None
    if args.sample:
        if args.prod:
            # Production mode: use fixed seed for reproducibility
            seed = 42
        else:
            # Debug mode: use random seed
            seed = random.randint(0, 2**32 - 1)

        random.seed(seed)
        print(f"Random seed: {seed}")

        if args.sample < len(raw_data):
            print(f"Sampling {args.sample} conversations...")
            raw_data = random.sample(raw_data, args.sample)

    # Process conversations
    print(f"\nProcessing {len(raw_data)} conversations...")
    processed_conversations = []

    for i, conv in enumerate(raw_data, 1):
        if args.verbose:
            print(f"\n--- Processing {i}/{len(raw_data)}: {conv.get('title', 'Untitled')} ---")
        else:
            print(f"Processing {i}/{len(raw_data)}...", end='\r')

        try:
            # Add summaries
            processed = process_conversation_with_summaries(
                client,
                model,
                conv,
                verbose=args.verbose
            )

            # Extract memories if not disabled
            if not args.no_memories:
                if args.verbose:
                    print("Extracting memories...")
                memories = extract_memories(client, model, processed)
                processed['memories'] = memories

            processed_conversations.append(processed)

        except Exception as e:
            print(f"\nError processing conversation {conv.get('id')}: {e}")
            continue

    print(f"\n\nSuccessfully processed {len(processed_conversations)}/{len(raw_data)} conversations")

    # Create metadata
    metadata = create_metadata(model, llm_mode, seed=seed, n_samples=args.sample)

    # Save results
    print(f"\nSaving to {output_file}...")
    save_conversations(processed_conversations, output_file, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
