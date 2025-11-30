import json
from datetime import datetime
from typing import List, Dict, Any, Optional


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

            # Separate text content from image/attachment parts
            text_content = ""
            images = []

            if isinstance(parts, list) and len(parts) > 0:
                for part in parts:
                    if isinstance(part, str):
                        # Text part
                        text_content += part
                    elif isinstance(part, dict):
                        # Non-text part (image, file, etc.)
                        # ChatGPT images typically have content_type starting with 'image_asset_pointer'
                        # or have 'asset_pointer' or 'image_url' fields
                        content_type = part.get('content_type', '')
                        if 'image' in content_type.lower() or 'asset_pointer' in part:
                            image_info = {
                                'content_type': content_type,
                            }
                            # Extract various possible image identifiers
                            if 'asset_pointer' in part:
                                image_info['asset_pointer'] = part['asset_pointer']
                            if 'image_url' in part:
                                image_info['image_url'] = part['image_url']
                            if 'metadata' in part:
                                image_info['metadata'] = part['metadata']
                            images.append(image_info)

            # Only add message if it has text or images
            if text_content or images:
                timestamp = message_data.get('create_time')
                readable_time = None
                if timestamp:
                    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                metadata = message_data.get('metadata', {})

                clean_msg = {
                    "role": message_data['author']['role'],
                    "text": text_content,
                    "timestamp": readable_time,
                    "model_slug": metadata.get('model_slug')
                }

                # Add images field if there are any
                if images:
                    clean_msg["images"] = images

                # Add additional metadata fields if present
                if 'finish_details' in metadata:
                    finish_details = metadata['finish_details']
                    if 'type' in finish_details:
                        clean_msg["finish_reason"] = finish_details['type']

                if 'weight' in metadata:
                    clean_msg["weight"] = metadata['weight']

                if 'end_turn' in metadata:
                    clean_msg["end_turn"] = metadata['end_turn']

                if 'recipient' in metadata:
                    clean_msg["recipient"] = metadata['recipient']

                # Citations (from web browsing)
                if 'citations' in metadata:
                    clean_msg["citations"] = metadata['citations']

                # Command/tool invocations
                if 'command' in metadata:
                    clean_msg["command"] = metadata['command']

                messages.append(clean_msg)

        # Move pointer to the parent
        current_node_id = node.get('parent')

    # The loop extracted them in reverse (Newest -> Oldest), so flip it back
    return messages[::-1]


def flatten_turn(turn: Dict[str, str], agent_mode: str = 'summarised_conversation') -> str:
    """Flattens a turn dictionary to a formatted string for LLM input."""
    for key, text in turn.items():
        if key == 'user':
            return f"[user]\n{text}\n"
        if key == agent_mode:
            return f"[agent]\n{text}\n"
    return ''
