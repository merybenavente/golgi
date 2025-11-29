"""Simple web viewer for conversation summaries."""
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Global state
current_conversations = []
current_file = None
current_metadata = None

def find_eligible_files():
    """Find all eligible conversation files in data/ directory."""
    data_dir = Path(__file__).parent / "data"
    eligible = []

    for subdir in ['debug', 'prod']:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            for file in subdir_path.glob('*.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Check if it's a valid conversations file with memories
                    if isinstance(data, dict) and 'conversations' in data:
                        convs = data['conversations']
                    elif isinstance(data, list):
                        convs = data
                    else:
                        continue

                    # Verify it has the right structure with memories
                    if convs and isinstance(convs, list) and len(convs) > 0:
                        # Check if conversations have both 'turns' AND 'memories'
                        has_valid_structure = all(
                            isinstance(c, dict) and
                            'turns' in c and
                            'memories' in c
                            for c in convs[:min(3, len(convs))]
                        )
                        if has_valid_structure:
                            relative_path = str(file.relative_to(data_dir))
                            eligible.append({
                                'path': relative_path,
                                'name': file.name,
                                'full_path': str(file)
                            })
                except (json.JSONDecodeError, IOError):
                    continue

    return sorted(eligible, key=lambda x: x['name'], reverse=True)

def load_conversations_from_file(file_path):
    """Load conversations from a specific JSON file, returns (conversations, metadata)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both wrapped and unwrapped formats
    if isinstance(data, dict) and 'conversations' in data:
        return data['conversations'], data.get('metadata')
    elif isinstance(data, list):
        return data, None
    else:
        return [], None

def get_default_file():
    """Get the default file to load."""
    files = find_eligible_files()
    if files:
        return files[0]['full_path']
    # Fallback to old hardcoded path
    return str(Path(__file__).parent / "data" / "debug" / "summarised_conversation_memories.json")

# Initialize with default file
current_file = get_default_file()
current_conversations, current_metadata = load_conversations_from_file(current_file)

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html',
                         total_conversations=len(current_conversations),
                         current_file=Path(current_file).name,
                         has_metadata=current_metadata is not None)

@app.route('/api/conversation/<int:index>')
def get_conversation(index):
    """Get conversation by index."""
    if 0 <= index < len(current_conversations):
        return jsonify(current_conversations[index])
    return jsonify({"error": "Index out of range"}), 404

@app.route('/api/files')
def get_files():
    """Get list of available conversation files."""
    files = find_eligible_files()
    current_name = Path(current_file).relative_to(Path(__file__).parent / "data")
    return jsonify({
        'files': files,
        'current': str(current_name)
    })

@app.route('/api/metadata')
def get_metadata():
    """Get metadata for the currently loaded file."""
    if current_metadata:
        return jsonify(current_metadata)
    return jsonify({"error": "No metadata available"}), 404

@app.route('/api/load-file', methods=['POST'])
def load_file():
    """Load a different conversation file."""
    global current_conversations, current_file, current_metadata

    data = request.json
    file_path = data.get('file_path')

    if not file_path:
        return jsonify({"error": "No file path provided"}), 400

    full_path = Path(__file__).parent / "data" / file_path

    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        current_conversations, current_metadata = load_conversations_from_file(full_path)
        current_file = str(full_path)
        return jsonify({
            "success": True,
            "total_conversations": len(current_conversations),
            "file": file_path,
            "has_metadata": current_metadata is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
