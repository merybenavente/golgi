"""LLM-based processing (summarization and memory extraction)."""
from typing import Dict, Any
from openai import OpenAI

from .config import SUMMARIZATION_PROMPT, MEMORY_EXTRACTION_PROMPT
from .conversation import extract_linear_conversation, flatten_turn


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
