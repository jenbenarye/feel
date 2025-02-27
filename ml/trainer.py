import os
import torch
from dataclasses import dataclass
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format
from kto_dataset_processor import process_feel_dataset, SupportedLanguages
from datetime import datetime
import wandb
from enum import Enum
from typing import Optional
from pathlib import Path


# PEFT library: attach and load adapters
from peft import get_peft_model, PeftModel

####################################
#  CONFIGURATION
####################################


@dataclass
class ScriptArguments:
    """
    Configuration for the script.
    """
    process_dataset_func: callable = process_feel_dataset
    checkpoint_path: str = None
    push_to_hub: bool = True
    language: str = "English"  # Default to English

    def __post_init__(self):
        """Validate the language after initialization"""
        try:
            # This will raise ValueError if language is not in the enum
            SupportedLanguages(self.language)
        except ValueError:
            supported_langs = "\n- ".join([lang.value for lang in SupportedLanguages])
            raise ValueError(
                f"Invalid language: '{self.language}'\n"
                f"Supported languages are:\n- {supported_langs}"
            )

@dataclass
class ModelArguments(ModelConfig):
    """
    Configuration for the model.
    """
    model_name: str = "CohereForAI/aya-expanse-8b"
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16
    trust_remote_code: bool = True

@dataclass
class TrainingArguments(KTOConfig):
    """
    Configuration for the KTO trainer.
    """
    output_dir: str = f"kto_{ModelArguments.model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    eval_steps: int = 500
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_first_step: bool = True

# Initialize configurations
script_args = ScriptArguments()
training_args = TrainingArguments()
model_args = ModelArguments()

####################################
#  HELPER FUNCTIONS
####################################

def load_model_and_tokenizer(model_args):
    """
    Load the base model and tokenizer from the Hugging Face Hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code
    )

    # Set pad token if it is missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup chat format if not available on the tokenizer
    if not getattr(tokenizer, "chat_template", None):
        model, tokenizer = setup_chat_format(model, tokenizer)

    return model, tokenizer

def get_adapter_path(model_name: str, language: str, timestamp: str = None) -> Path:
    """
    Generate standardized adapter path.
    If timestamp is None, returns the base language directory.
    Otherwise, returns specific adapter version path.

    Format: adapters/{model_name}/{language}/version_{timestamp}
    """
    # Clean model name (remove slashes, etc.)
    clean_model_name = model_name.replace('/', '_')

    base_path = Path("adapters") / clean_model_name / language
    if timestamp:
        return base_path / f"version_{timestamp}"
    return base_path

def load_latest_adapter(model, model_name: str, language: str) -> tuple[PeftModel, str]:
    """
    Load the most recent adapter for given model and language.
    Returns: (loaded_model, timestamp of loaded adapter)
    """
    adapter_base = get_adapter_path(model_name, language)

    if not adapter_base.exists():
        return None, None

    # Get all version directories and sort by timestamp
    versions = sorted(
        [d for d in adapter_base.glob("version_*")],
        key=lambda x: x.name,
        reverse=True
    )

    if not versions:
        return None, None

    latest_version = versions[0]
    timestamp = latest_version.name.replace("version_", "")

    model = PeftModel.from_pretrained(model, latest_version, is_trainable=True)
    return model, timestamp

####################################
#  MAIN LOGIC
####################################

def main():
    # Initialize wandb for logging
    wandb.init(project="kto")

    # Get timestamp at start of training
    training_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args)
    ref_model, _ = load_model_and_tokenizer(model_args)
    print("Models and tokenizer loaded.")

    # Load existing adapter or create new one
    loaded_model, previous_timestamp = load_latest_adapter(
        model,
        model_args.model_name,
        script_args.language
    )

    if loaded_model is not None:
        model = loaded_model
        print(f"Loaded existing adapter trained at {previous_timestamp}")
    else:
        # Initialize new LoRA adapter
        peft_config = get_peft_config(model_args)
        model = get_peft_model(model, peft_config)
        print("Initialized new adapter")

    # -----------------------------
    # Data Preparation and Training
    # -----------------------------
    print("Processing dataset...")
    dataset = script_args.process_dataset_func(script_args.language)
    print("Dataset processed.")

    print("Initializing trainer...")
    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
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

    # Save the adapter
    adapter_path = get_adapter_path(
        model_args.model_name,
        script_args.language,
        training_timestamp
    )
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving adapter to: {adapter_path}")
    model.save_pretrained(adapter_path)

    # Save metadata
    metadata = AdapterMetadata(
        training_timestamp=training_timestamp,
        dataset_entries=[entry["id"] for entry in dataset],
        training_params={
            "max_weight": script_args.max_weight,
            "min_weight": script_args.min_weight,
            "decay_factor": script_args.decay_factor,
            "training_mode": script_args.training_mode
        },
        model_name=model_args.model_name,
        language=script_args.language,
        version=training_timestamp
    )
    metadata.save(adapter_path / "metadata.json")

    if script_args.push_to_hub:
        # Using a consistent naming pattern that links to the FEEL project
        repo_id = f"feel-fl/kto-lora-adapter-{script_args.language}"
        print(f"Pushing adapter to Hugging Face Hub at {repo_id}...")
        model.push_to_hub(repo_id=repo_id)

    print("Process completed.")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
