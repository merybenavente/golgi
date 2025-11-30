"""Tests for storage operations."""
import os
import json
import tempfile
import pytest
from chatgpt_parser.storage import (
    get_git_commit,
    create_metadata,
    save_conversations,
    load_processed_conversations,
    generate_output_filename
)


def test_get_git_commit():
    """Test git commit hash retrieval."""
    commit = get_git_commit()

    # Should return either a valid hash or 'unknown'
    assert isinstance(commit, str)
    assert len(commit) > 0
    # If in git repo, commit hash is 40 chars; otherwise 'unknown'
    assert commit == 'unknown' or len(commit) == 40


def test_create_metadata():
    """Test metadata creation."""
    metadata = create_metadata(
        model="gpt-4o-mini",
        llm_mode="openai",
        seed=42,
        n_samples=10
    )

    assert metadata["model"] == "gpt-4o-mini"
    assert metadata["llm_mode"] == "openai"
    assert metadata["random_seed"] == 42
    assert metadata["n_samples"] == 10
    assert "processing_timestamp" in metadata
    assert "git_commit" in metadata
    assert "prompts" in metadata
    assert "summarization" in metadata["prompts"]
    assert "memory_extraction" in metadata["prompts"]


def test_create_metadata_minimal():
    """Test metadata creation without optional fields."""
    metadata = create_metadata(
        model="llama3.1:8b",
        llm_mode="local"
    )

    assert metadata["model"] == "llama3.1:8b"
    assert metadata["llm_mode"] == "local"
    assert "random_seed" not in metadata
    assert "n_samples" not in metadata


def test_save_and_load_conversations():
    """Test saving and loading conversation cycle."""
    test_conversations = [
        {
            "id": "test1",
            "title": "Test Conversation",
            "turns": [
                {"user": "Hello"},
                {"assistant": "Hi!", "summarised_conversation": "Greeted user"}
            ],
            "memories": "- User prefers informal greetings"
        }
    ]

    test_metadata = {
        "model": "gpt-4o-mini",
        "llm_mode": "openai",
        "git_commit": "abc123"
    }

    # Use temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name

    try:
        # Save
        save_conversations(test_conversations, temp_filename, test_metadata)

        # Verify file exists
        assert os.path.exists(temp_filename)

        # Load
        loaded_convs, loaded_meta = load_processed_conversations(temp_filename)

        # Verify content
        assert len(loaded_convs) == 1
        assert loaded_convs[0]["id"] == "test1"
        assert loaded_convs[0]["title"] == "Test Conversation"
        assert loaded_meta["model"] == "gpt-4o-mini"
        assert loaded_meta["git_commit"] == "abc123"

    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_load_conversations_missing_file():
    """Test loading from non-existent file."""
    convs, meta = load_processed_conversations("nonexistent_xyz.json")

    assert convs == []
    assert meta is None


def test_load_conversations_legacy_format():
    """Test loading legacy format (direct list without metadata wrapper)."""
    test_data = [
        {"id": "1", "title": "Test"}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
        json.dump(test_data, f)

    try:
        convs, meta = load_processed_conversations(temp_filename)

        assert len(convs) == 1
        assert convs[0]["id"] == "1"
        assert meta is None  # No metadata in legacy format

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_generate_output_filename_debug():
    """Test output filename generation for debug mode."""
    filename = generate_output_filename("gpt-4o-mini", n_samples=5, is_prod=False)

    assert "data/debug/" in filename
    assert "convs_with_memories_" in filename
    assert "gpt4omini" in filename  # Special chars removed
    assert "nsamples5" in filename
    assert filename.endswith(".json")


def test_generate_output_filename_prod():
    """Test output filename generation for production mode."""
    filename = generate_output_filename("llama3.1:8b", n_samples=None, is_prod=True)

    assert "data/prod/" in filename
    assert "convs_with_memories_" in filename
    assert "llama318b" in filename  # Special chars removed
    assert "nsamples" not in filename  # No sample count
    assert filename.endswith(".json")


def test_generate_output_filename_creates_directory():
    """Test that generate_output_filename creates directory if missing."""
    # Use temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a subdirectory path that doesn't exist yet
        test_dir = os.path.join(tmpdir, "test_data", "debug")

        # Temporarily patch the function to use our test directory
        import chatgpt_parser.storage as storage_module
        original_generate = storage_module.generate_output_filename

        def patched_generate(model_name, n_samples, is_prod):
            # Use test directory instead of "data/"
            date_str = storage_module.datetime.now().strftime('%m%d')
            clean_model = model_name.replace(':', '').replace('.', '').replace('-', '')
            parts = [f"convs_with_memories_{date_str}_{clean_model}"]
            if n_samples:
                parts.append(f"nsamples{n_samples}")
            directory = os.path.join(tmpdir, "test_data", "prod" if is_prod else "debug")
            os.makedirs(directory, exist_ok=True)
            filename = "_".join(parts) + ".json"
            return os.path.join(directory, filename)

        try:
            storage_module.generate_output_filename = patched_generate
            filename = storage_module.generate_output_filename("test-model", None, False)

            # Directory should now exist
            assert os.path.exists(os.path.dirname(filename))

        finally:
            storage_module.generate_output_filename = original_generate
