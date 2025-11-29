"""Simple web viewer for conversation summaries."""
import json
from pathlib import Path
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Load conversation data
DATA_FILE = Path(__file__).parent / "data" / "debug" / "summarised_conversation_memories.json"

def load_conversations():
    """Load conversations from JSON file."""
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

conversations = load_conversations()

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html', total_conversations=len(conversations))

@app.route('/api/conversation/<int:index>')
def get_conversation(index):
    """Get conversation by index."""
    if 0 <= index < len(conversations):
        return jsonify(conversations[index])
    return jsonify({"error": "Index out of range"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
