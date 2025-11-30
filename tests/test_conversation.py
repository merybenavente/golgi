"""Tests for conversation parsing and extraction."""
import os
import json
import pytest
from chatgpt_parser.conversation import (
    load_conversations,
    extract_linear_conversation,
    filter_conversations,
    flatten_turn
)


def test_load_conversations_valid_file():
    """Test loading conversations from existing data file."""
    file_path = "data/conversations.json"

    if not os.path.exists(file_path):
        pytest.skip(f"Test data file not found: {file_path}")

    conversations = load_conversations(file_path)

    assert isinstance(conversations, list)
    assert len(conversations) > 0
    assert 'mapping' in conversations[0]
    assert 'current_node' in conversations[0]


def test_load_conversations_missing_file():
    """Test handling of missing file."""
    conversations = load_conversations("nonexistent_file.json")

    assert isinstance(conversations, list)
    assert len(conversations) == 0


def test_extract_linear_conversation():
    """Test extraction of linear conversation from graph structure."""
    # Create minimal test conversation with graph structure
    test_conv = {
        "mapping": {
            "root": {
                "message": None,
                "parent": None
            },
            "msg1": {
                "message": {
                    "content": {"parts": ["Hello"]},
                    "author": {"role": "user"},
                    "create_time": 1234567890
                },
                "parent": "root"
            },
            "msg2": {
                "message": {
                    "content": {"parts": ["Hi there!"]},
                    "author": {"role": "assistant"},
                    "create_time": 1234567891,
                    "metadata": {"model_slug": "gpt-4"}
                },
                "parent": "msg1"
            }
        },
        "current_node": "msg2"
    }

    messages = extract_linear_conversation(test_conv)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["text"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["text"] == "Hi there!"
    assert messages[1]["model_slug"] == "gpt-4"


def test_filter_conversations_by_keyword():
    """Test filtering conversations by keyword in title."""
    test_convs = [
        {"id": "1", "title": "Python coding help", "mapping": {}, "current_node": None},
        {"id": "2", "title": "Recipe for pasta", "mapping": {}, "current_node": None},
        {"id": "3", "title": "Python debugging", "mapping": {}, "current_node": None}
    ]

    filtered = filter_conversations(test_convs, keyword="python")

    assert len(filtered) == 2
    assert all("python" in conv["title"].lower() for conv in filtered)


def test_filter_conversations_by_min_turns():
    """Test filtering conversations by minimum turns."""
    test_convs = [
        {
            "id": "1",
            "title": "Short chat",
            "mapping": {
                "root": {"message": None, "parent": None},
                "msg1": {
                    "message": {
                        "content": {"parts": ["Hi"]},
                        "author": {"role": "user"},
                        "create_time": 1234567890
                    },
                    "parent": "root"
                }
            },
            "current_node": "msg1"
        },
        {
            "id": "2",
            "title": "Long chat",
            "mapping": {
                "root": {"message": None, "parent": None},
                "msg1": {
                    "message": {
                        "content": {"parts": ["Hello"]},
                        "author": {"role": "user"},
                        "create_time": 1234567890
                    },
                    "parent": "root"
                },
                "msg2": {
                    "message": {
                        "content": {"parts": ["Hi!"]},
                        "author": {"role": "assistant"},
                        "create_time": 1234567891
                    },
                    "parent": "msg1"
                },
                "msg3": {
                    "message": {
                        "content": {"parts": ["How are you?"]},
                        "author": {"role": "user"},
                        "create_time": 1234567892
                    },
                    "parent": "msg2"
                }
            },
            "current_node": "msg3"
        }
    ]

    filtered = filter_conversations(test_convs, min_turns=2)

    assert len(filtered) == 1
    assert filtered[0]["id"] == "2"


def test_flatten_turn_user():
    """Test flattening user turn."""
    turn = {"user": "What is Python?"}

    result = flatten_turn(turn)

    assert result == "[user]\nWhat is Python?\n"


def test_flatten_turn_agent():
    """Test flattening agent turn."""
    turn = {"summarised_conversation": "Explained Python basics"}

    result = flatten_turn(turn)

    assert result == "[agent]\nExplained Python basics\n"


def test_flatten_turn_empty():
    """Test flattening empty turn."""
    turn = {"other_key": "value"}

    result = flatten_turn(turn)

    assert result == ''


def test_extract_conversation_with_images():
    """Test extraction of conversation with image attachments."""
    test_conv = {
        "mapping": {
            "root": {
                "message": None,
                "parent": None
            },
            "msg1": {
                "message": {
                    "content": {
                        "parts": [
                            "Look at this image:",
                            {
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "file-abc123",
                                "metadata": {
                                    "width": 1024,
                                    "height": 768
                                }
                            }
                        ]
                    },
                    "author": {"role": "user"},
                    "create_time": 1234567890
                },
                "parent": "root"
            },
            "msg2": {
                "message": {
                    "content": {"parts": ["I can see the image!"]},
                    "author": {"role": "assistant"},
                    "create_time": 1234567891
                },
                "parent": "msg1"
            }
        },
        "current_node": "msg2"
    }

    messages = extract_linear_conversation(test_conv)

    assert len(messages) == 2

    # Check first message (with image)
    msg_with_image = messages[0]
    assert msg_with_image["role"] == "user"
    assert msg_with_image["text"] == "Look at this image:"
    assert "images" in msg_with_image
    assert len(msg_with_image["images"]) == 1

    # Check image data
    image = msg_with_image["images"][0]
    assert image["content_type"] == "image_asset_pointer"
    assert image["asset_pointer"] == "file-abc123"
    assert image["metadata"]["width"] == 1024
    assert image["metadata"]["height"] == 768

    # Check second message (no images)
    msg_without_image = messages[1]
    assert msg_without_image["role"] == "assistant"
    assert msg_without_image["text"] == "I can see the image!"
    assert "images" not in msg_without_image


def test_extract_conversation_image_only():
    """Test extraction of message with only image, no text."""
    test_conv = {
        "mapping": {
            "root": {
                "message": None,
                "parent": None
            },
            "msg1": {
                "message": {
                    "content": {
                        "parts": [
                            {
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "file-xyz789",
                                "image_url": "https://example.com/image.png"
                            }
                        ]
                    },
                    "author": {"role": "user"},
                    "create_time": 1234567890
                },
                "parent": "root"
            }
        },
        "current_node": "msg1"
    }

    messages = extract_linear_conversation(test_conv)

    assert len(messages) == 1

    msg = messages[0]
    assert msg["role"] == "user"
    assert msg["text"] == ""  # No text, only image
    assert "images" in msg
    assert len(msg["images"]) == 1
    assert msg["images"][0]["asset_pointer"] == "file-xyz789"
    assert msg["images"][0]["image_url"] == "https://example.com/image.png"
