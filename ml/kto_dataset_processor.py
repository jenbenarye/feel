from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from ipdb import set_trace as st
import tiktoken
from transformers import AutoTokenizer

def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text using model's tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))

def format_conversation(messages: list, model_name: str) -> str:
    """Format messages using model's chat template"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.apply_chat_template(messages, tokenize=False)

def transform_conversation(
    entry: dict,
    model_name: str,
    max_history_turns: int = 10,
    max_history_tokens: int = 4000
) -> list:
    """Transform conversation into KTO format with history"""
    data_points = []
    conversation = entry["conversation"]

    for i, message in enumerate(conversation):
        # Only process assistant messages with ratings
        if message["role"] != "assistant" or message["rating"] not in [1, -1]:
            continue

        # Get previous messages up to limits
        history = []
        tokens = 0
        turns = 0

        # Start from i-1 instead of going through all previous messages
        for prev in reversed(conversation[max(0, i-1):i]):
            if turns >= max_history_turns:
                break

            history.insert(0, prev)
            formatted = format_conversation(history, model_name)
            tokens = count_tokens(formatted, model_name)

            if tokens > max_history_tokens:
                history.pop(0)
                break

            turns += 1

        # Format prompt with just the immediate previous message
        prompt = format_conversation([conversation[i-1]], model_name) if i > 0 else ""

        data_points.append({
            "prompt": prompt.strip(),
            "completion": message["content"].strip(),
            "label": message["rating"] == 1,
            "timestamp": entry["timestamp"],
            "session_id": entry["session_id"],
            "conversation_id": entry["conversation_id"]
        })

    return data_points

def process_feel_dataset(
    model_name: str = "HuggingFaceH4/zephyr-7b-beta",
    max_history_turns: int = 10,
    max_history_tokens: int = 4000
):
    """
    Processes the feel dataset into a format suitable for KTO training using TRL.

    Args:
        model_name: Name of the model to format for
        max_history_turns: Maximum number of previous turns to include in history
        max_history_tokens: Maximum number of tokens allowed in history

    Returns:
        dict: A dictionary containing the 'train' and 'test' splits of the dataset in KTO format
    """
    # Load feel dataset from HuggingFace
    feel_dataset = load_dataset("feel-fl/feel-feedback")["train"]
    kto_data = []

    # Process all conversations in the dataset
    for entry in feel_dataset:
        kto_data.extend(transform_conversation(
            entry,
            model_name,
            max_history_turns,
            max_history_tokens
        ))

    # Convert to DataFrame
    kto_df = pd.DataFrame(kto_data)

    # Split into train and test sets (70% train, 30% test)
    train_df, test_df = train_test_split(kto_df, test_size=0.3, random_state=42)

    # Reset index to remove '__index_level_0__'
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return {"train": train_dataset, "test": test_dataset}

if __name__ == "__main__":
    # Process the dataset
    datasets = process_feel_dataset()

    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Train set size: {len(datasets['train'])}")
    print(f"Test set size: {len(datasets['test'])}")

    # Print distribution of positive/negative labels
    train_labels = datasets['train']['label']
    test_labels = datasets['test']['label']

    print("\nLabel Distribution:")
    print("Train set:")
    print(f"Positive feedback: {sum(train_labels)}")
    print(f"Negative feedback: {len(train_labels) - sum(train_labels)}")
    print(f"Positive ratio: {sum(train_labels)/len(train_labels):.2%}")

    print("\nTest set:")
    print(f"Positive feedback: {sum(test_labels)}")
    print(f"Negative feedback: {len(test_labels) - sum(test_labels)}")
    print(f"Positive ratio: {sum(test_labels)/len(test_labels):.2%}")

    # Load original FEEL dataset
    feel_dataset = load_dataset("feel-fl/feel-feedback", split="train")

    # Print one original conversation
    print("\nOriginal conversation from FEEL dataset:")
    print(json.dumps(feel_dataset[0], indent=2))

    # Print sample entries from processed dataset
    print("\nSample entries from processed KTO dataset:")
    print("\n" + "="*80 + "\nTRAIN SET SAMPLES\n" + "="*80)

    # for i, example in enumerate(datasets['train'].select(range(min(3, len(datasets['train']))))):
    #     print(f"\nEntry #{i+1}:")
    #     print("-" * 40)
    #     for field, value in example.items():
    #         print(f"\n{field}:")
    #         if isinstance(value, str):
    #             # Print strings with line breaks for better readability
    #             print(f"{value}")
    #         else:
    #             print(f"{value}")
    #     print("\n" + "-"*80)

    # print("\n" + "="*80 + "\nTEST SET SAMPLES\n" + "="*80)

    # for i, example in enumerate(datasets['test'].select(range(min(3, len(datasets['test']))))):
    #     print(f"\nEntry #{i+1}:")
    #     print("-" * 40)
    #     for field, value in example.items():
    #         print(f"\n{field}:")
    #         if isinstance(value, str):
    #             # Print strings with line breaks for better readability
    #             print(f"{value}")
    #         else:
    #             print(f"{value}")
    #     print("\n" + "-"*80)
