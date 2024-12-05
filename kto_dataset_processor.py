from datasets import load_dataset, Dataset
import pandas as pd
from pdb import set_trace as st


def process_dataset_ultrafeedback():
    """
    Processes the 'HuggingFaceH4/ultrafeedback_binarized' dataset into a unified train and test split.

    Returns:
        dict: A dictionary containing the unified 'train' and 'test' splits of the dataset in the KTO format.
              Each split is a Hugging Face Dataset object.
    """
    # Load the dataset
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    dataset = load_dataset(dataset_name)

    # Function to transform a single example into the desired schema
    def transform_data(example):
        data_points = []
        # Chosen completion
        chosen_completion = example["chosen"][1]["content"]
        data_points.append({
            "prompt": example["prompt"],
            "completion": chosen_completion.strip(),
            "label": True
        })
        # Rejected completion
        rejected_completion = example["rejected"][1]["content"]
        data_points.append({
            "prompt": example["prompt"],
            "completion": rejected_completion.strip(),
            "label": False
        })
        return data_points

    # Combine splits into unified train and test sets
    train_data = []
    test_data = []

    for split_name, split_data in dataset.items():
        if "train" in split_name:
            for example in split_data:
                train_data.extend(transform_data(example))
        elif "test" in split_name:
            for example in split_data:
                test_data.extend(transform_data(example))

    # Convert unified data to Hugging Face Dataset
    unified_train = Dataset.from_pandas(pd.DataFrame(train_data))
    unified_test = Dataset.from_pandas(pd.DataFrame(test_data))

    return {"train": unified_train, "test": unified_test}


# if __name__ == "__main__":
#     kto_dataset = process_dataset_ultrafeedback()
#     st()
