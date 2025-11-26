import json
from datetime import datetime

def load_conversations(file_path):
    """
    Loads the raw conversations.json file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

def extract_linear_conversation(conversation_data):
    """
    Traverses the conversation graph backwards from the 'current_node' 
    to reconstruct the linear chat history.
    """
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

def filter_conversations(all_conversations, keyword=None, min_turns=0):
    """
    Filters the raw list of conversations based on keywords or length.
    Returns a list of CLEANED linear objects.
    """
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

# --- Quick Test Block (will run if you execute this file directly) ---
if __name__ == "__main__":
    # 1. Update this path to your actual file
    FILE_PATH = "conversations.json" 
    
    print("Loading data...")
    raw_data = load_conversations(FILE_PATH)
    
    if raw_data:
        # Example: Find chats about "travel" with at least 4 messages
        print("Filtering...")
        clean_chats = filter_conversations(raw_data, keyword="travel", min_turns=4)
        
        if clean_chats:
            sample = clean_chats[0]
            print(f"Found {len(clean_chats)} conversations.")
            print(f"--- Sample: {sample['title']} ---")
            for msg in sample['messages']:
                print(f"[{msg['role']} @ {msg['timestamp']}]: {msg['text'][:50]}...")
        else:
            print("No conversations found matching criteria.")
