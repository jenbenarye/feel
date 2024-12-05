import torch
from dataclasses import dataclass
from accelerate import PartialState
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format
from dataloaders.data_loader import get_oasst
from pdb import set_trace as st
import wandb


####################################
#  CONFIGURATION
####################################

@dataclass
class ScriptArguments:
    """
    Configuration for the script.
    """
    dataset_name: str = "OpenAssistant/oasst1"  # Dataset name or path
    output_dir: str = "/raid/lingo/jen_ben/HF-RLHF/kto_nov_24_2_epochs"  # Output directory
    pretrained_model_name: str = "mistralai/Mistral-7B-v0.1"  # Pretrained model name or path
    checkpoint_path: str = "/raid/lingo/jen_ben/HF-RLHF/kto_nov_24_2_epochs"  # Checkpoint path
    push_to_hub: bool = False  # Whether to push the model to the Hugging Face hub

@dataclass
class TrainingArguments(KTOConfig):
    """
    Configuration for the KTO trainer.
    """
    output_dir: str = "/raid/lingo/jen_ben/HF-RLHF/kto_nov_24_2_epochs"
    num_train_epochs: int = 2 # did 1 epoch, then maybe try 2 epochs
    per_device_train_batch_size: int = 4 # 4 is the highes that runs well.
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    eval_steps: int = 500
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_first_step: bool = True

@dataclass
class ModelArguments(ModelConfig):
    """
    Configuration for the model.
    """
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16

# Initialize configurations
script_args = ScriptArguments()
training_args = TrainingArguments(output_dir=script_args.output_dir)
model_args = ModelArguments(model_name_or_path=script_args.pretrained_model_name)

####################################
#  HELPER FUNCTIONS
####################################

def load_model_and_tokenizer(model_args):
    """
    Load a model and tokenizer from a specified path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup chat format if not present
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    return model, tokenizer



def load_and_format_oasst_dataset(tokenizer):
    """
    Load, process, and format the OpenAssistant dataset into DPO-compatible format.

    Args:
        split (str): The dataset split to load ('train' or 'test').
        tokenizer (AutoTokenizer): Tokenizer to apply chat templates.
        num_proc (int, optional): Number of processes for parallel processing.

    Returns:
        Dataset: Processed and formatted dataset.
    """

    # Load oasst dataset
    train_dataset = get_oasst(split='train')

    # Initialize lists for DPO dataset
    dpo_train_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    # Process the dataset
    for prompt, key in train_dataset.data.items():  # Iterate over dataset
        if hasattr(key, "pairs") and key.pairs:  # Check if pairs exist
            for i, j in key.pairs:  # Process each preference pair
                # Add prompt and corresponding chosen/rejected completions
                dpo_train_data["prompt"].append(key.prompt)
                dpo_train_data["chosen"].append(key.generations[i])  # Chosen generation
                dpo_train_data["rejected"].append(key.generations[j])  # Rejected generation

    # Convert DPO data into a Dataset
    dpo_train_dataset = Dataset.from_dict(dpo_train_data)

    # Wrap it in a DatasetDict
    dataset_dict = DatasetDict({
        "train": dpo_train_dataset
    })


    test_dataset = get_oasst(split='test')

    dpo_test_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for prompt, key in test_dataset.data.items():  # Iterate over dataset
        if hasattr(key, "pairs") and key.pairs:  # Check if pairs exist
            for i, j in key.pairs:  # Process each preference pair
                # Add prompt and corresponding chosen/rejected completions
                dpo_test_data["prompt"].append(key.prompt)
                dpo_test_data["chosen"].append(key.generations[i])  # Chosen generation
                dpo_test_data["rejected"].append(key.generations[j])  # Rejected generation

    dpo_test_dataset = Dataset.from_dict(dpo_test_data)
    dataset_dict["test"] = dpo_test_dataset

    # If needed, reformat a DPO-formatted dataset (prompt, chosen, rejected) to a KTO-format (prompt, completion, label)
    dataset_dict = maybe_unpair_preference_dataset(dataset_dict, num_proc=training_args.dataset_num_proc)
    print(f'loaded dataset')


    # Apply chat template
    def format_dataset(example):
        # Ensure prompt is in the correct structure
        if isinstance(example["prompt"], str):
            example["prompt"] = [{"role": "user", "content": example["prompt"]}]
        elif isinstance(example["prompt"], list):
            # If it's already a list, ensure each element has the "role" and "content" keys
            for item in example["prompt"]:
                if "role" not in item or "content" not in item:
                    raise ValueError(f"Each item in 'prompt' must have 'role' and 'content': {item}")

        # Ensure completion is in the correct structure
        if isinstance(example["completion"], str):
            example["completion"] = [{"role": "assistant", "content": example["completion"]}]  # Wrap as a list of dictionaries
        elif isinstance(example["completion"], list):
            # If it's already a list, ensure each element has the "role" and "content" keys
            for item in example["completion"]:
                if "role" not in item or "content" not in item:
                    raise ValueError(f"Each item in 'completion' must have 'role' and 'content': {item}")

        # Now apply the chat template
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)

        return example


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        dataset = dataset_dict.map(format_dataset, num_proc=training_args.dataset_num_proc)

    return dataset

####################################
#  MAIN LOGIC
####################################

def main():
    # Initialize wandb
    wandb.init(project="kto")

    # Load models and tokenizer
    print("Loading models and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args)
    ref_model, _ = load_model_and_tokenizer(model_args)
    print("Models and tokenizer loaded.")

    # Load and process datasets
    print("Loading, processing, and formatting dataset...")
    dataset = load_and_format_oasst_dataset(
        tokenizer=tokenizer,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Evaluation
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Log metrics to wandb
    wandb.log({
        "epoch": metrics.get("epoch"),
        "grad_norm": metrics.get("grad_norm"),
        "kl": metrics.get("kl"),
        "learning_rate": metrics.get("learning_rate"),
        "logits/chosen": metrics.get("logits/chosen"),
        "logits/rejected": metrics.get("logits/rejected"),
        "logps/chosen": metrics.get("logps/chosen"),
        "logps/rejected": metrics.get("logps/rejected"),
        "loss": metrics.get("loss"),
        "rewards/chosen": metrics.get("rewards/chosen"),
        "rewards/margins": metrics.get("rewards/margins"),
        "rewards/rejected": metrics.get("rewards/rejected"),
        "step": metrics.get("step")
    })

    # Save model and optionally push to hub
    trainer.save_model(training_args.output_dir)
    if script_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("Process completed.")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
