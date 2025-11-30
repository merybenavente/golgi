"""CLI and orchestration."""
import sys
import argparse
import random

from .llm_client import setup_llm_client
from .conversation import load_conversations
from .processor import process_conversation_with_summaries, extract_memories
from .storage import (
    generate_output_filename,
    create_metadata,
    save_conversations
)


def main() -> None:
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
