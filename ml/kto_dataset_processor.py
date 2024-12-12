from datasets import load_dataset, Dataset
import pandas as pd
from pdb import set_trace as st


def process_dataset_ultrafeedback():
    """
    Processes the 'train_prefs' and 'test_prefs' splits of the 'HuggingFaceH4/ultrafeedback_binarized' dataset
    into a unified format for preference modeling.

    Returns:
        dict: A dictionary containing the unified 'train' and 'test' splits of the dataset in the KTO format.
              Each split is a Hugging Face Dataset object.
    """
    # Load the relevant splits of the dataset
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    train_prefs = load_dataset(dataset_name, split="train_prefs")
    test_prefs = load_dataset(dataset_name, split="test_prefs")

    # Function to transform a single example into the desired schema
    def transform_data(example):
        data_points = []
        # Chosen completion
        chosen_completion = example["chosen"][1]["content"]
        if chosen_completion.strip():  # Check for non-empty completions
            data_points.append({
                "prompt": example["prompt"],
                "completion": chosen_completion.strip(),
                "label": True
            })
        # Rejected completion
        rejected_completion = example["rejected"][1]["content"]
        if rejected_completion.strip():  # Check for non-empty completions
            data_points.append({
                "prompt": example["prompt"],
                "completion": rejected_completion.strip(),
                "label": False
            })
        return data_points

    # Process train and test splits
    train_data = []
    test_data = []

    for example in train_prefs:
        train_data.extend(transform_data(example))

    for example in test_prefs:
        test_data.extend(transform_data(example))

    # Convert unified data to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)


    # Convert to Hugging Face Dataset
    unified_train = Dataset.from_pandas(train_df)
    unified_test = Dataset.from_pandas(test_df)

    return {"train": unified_train, "test": unified_test}


if __name__ == "__main__":
    kto_dataset = process_dataset_ultrafeedback()
    st()
