import torch
from dataclasses import dataclass
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format
from kto_dataset_processor import process_dataset_ultrafeedback
from datetime import datetime
import wandb

####################################
#  CONFIGURATION
####################################

@dataclass
class ScriptArguments:
    """
    Configuration for the script.
    """
    process_dataset_func: callable = process_dataset_ultrafeedback  # process_dataset function from kto_dataset_processor.py
    checkpoint_path: str = None  # Checkpoint path
    push_to_hub: bool = False  # Whether to push the model to the Hugging Face hub

@dataclass
class ModelArguments(ModelConfig):
    """
    Configuration for the model.
    """
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16

@dataclass
class TrainingArguments(KTOConfig):
    """
    Configuration for the KTO trainer.
    """
    output_dir: str = f"kto_{ModelArguments.model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4  # Highest that runs well
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
    Load a model and tokenizer from a specified path.
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

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup chat format if not present
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)



    return model, tokenizer


# def find_unknown_tokens(tokenizer, texts):
#     """
#     Identify tokens in the dataset that are not in the tokenizer's vocabulary.
#     """
#     all_tokens = set()
#     for text in texts:
#         tokens = tokenizer.tokenize(text)
#         all_tokens.update(tokens)
#     vocab = set(tokenizer.get_vocab().keys())
#     unknown_tokens = all_tokens - vocab
#     return unknown_tokens


# def add_tokens_to_tokenizer(tokenizer, model, dataset):
#     """
#     Extend the tokenizer's vocabulary with missing tokens and resize the model embeddings.
#     """
#     # Extract all texts from the dataset
#     texts = [example["completion"] for example in dataset["train"]]

#     # Identify unknown tokens
#     unknown_tokens = find_unknown_tokens(tokenizer, texts)
#     print(f"Found {len(unknown_tokens)} unknown tokens: {list(unknown_tokens)[:10]}...")

#     # Add unknown tokens to tokenizer
#     tokenizer.add_tokens(list(unknown_tokens))
#     model.resize_token_embeddings(len(tokenizer))
#     print(f"Tokenizer vocabulary size after extension: {len(tokenizer)}")


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

    # Load and process datasets using external function
    print("Processing dataset...")
    dataset = process_dataset_ultrafeedback()
    print("Dataset processed.")

    # # Extend tokenizer with missing tokens
    # print("Adding unknown tokens to tokenizer...")
    # add_tokens_to_tokenizer(tokenizer, model, dataset)
    # print("Tokenizer updated.")

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
        trainer.push_to_hub()

    print("Process completed.")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
