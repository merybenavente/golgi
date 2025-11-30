"""Integration smoke test."""
from unittest.mock import Mock, MagicMock
from chatgpt_parser.processor import (
    process_conversation_with_summaries,
    extract_memories
)


def test_end_to_end_processing_with_mock_llm():
    """Test complete conversation processing flow with mocked LLM."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Mocked summary"

    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    # Test conversation
    test_conversation = {
        "id": "test_conv_123",
        "title": "Test Conversation",
        "create_time": 1234567890,
        "update_time": 1234567900,
        "mapping": {
            "root": {
                "message": None,
                "parent": None
            },
            "msg1": {
                "message": {
                    "content": {"parts": ["What is Python?"]},
                    "author": {"role": "user"},
                    "create_time": 1234567890
                },
                "parent": "root"
            },
            "msg2": {
                "message": {
                    "content": {"parts": ["Python is a programming language."]},
                    "author": {"role": "assistant"},
                    "create_time": 1234567891,
                    "metadata": {"model_slug": "gpt-4"}
                },
                "parent": "msg1"
            }
        },
        "current_node": "msg2"
    }

    # Process conversation
    processed = process_conversation_with_summaries(
        client=mock_client,
        model="gpt-4",
        conversation=test_conversation,
        verbose=False
    )

    # Validate structure
    assert processed["id"] == "test_conv_123"
    assert processed["title"] == "Test Conversation"
    assert len(processed["turns"]) == 2

    # Check user turn
    user_turn = processed["turns"][0]
    assert "user" in user_turn
    assert user_turn["user"] == "What is Python?"

    # Check assistant turn (should have summary)
    assistant_turn = processed["turns"][1]
    assert "assistant" in assistant_turn
    assert "summarised_conversation" in assistant_turn
    assert assistant_turn["summarised_conversation"] == "Mocked summary"

    # Verify LLM was called
    assert mock_client.chat.completions.create.called


def test_memory_extraction_with_mock_llm():
    """Test memory extraction with mocked LLM."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "- User is learning Python\n- Interested in programming"

    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    # Processed conversation with turns
    processed_conversation = {
        "id": "test_123",
        "title": "Test",
        "turns": [
            {"user": "I want to learn Python"},
            {"assistant": "Great choice!", "summarised_conversation": "Encouraged user"}
        ]
    }

    # Extract memories
    memories = extract_memories(
        client=mock_client,
        model="gpt-4",
        conversation=processed_conversation
    )

    # Validate
    assert isinstance(memories, str)
    assert "Python" in memories
    assert "programming" in memories

    # Verify LLM was called
    assert mock_client.chat.completions.create.called
