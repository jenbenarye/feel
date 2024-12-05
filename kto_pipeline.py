import torch
from dataclasses import dataclass
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer, get_peft_config
from kto_dataset_processor import process_dataset_ultrafeedback
from datetime import datetime
import wandb

####################################
#  CONFIGURATION
####################################

@dataclass
class Config:
    """
    Configuration for the script.
    """
    # Dataset settings
    process_dataset_func: callable = process_dataset_ultrafeedback  # Dataset processing function

    # Model settings
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"  # Pretrained model name or path
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16

    # Training settings
    output_dir: str = f"kto_{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

    # Checkpoint and hub settings
    checkpoint_path: str = None
    push_to_hub: bool = False

# Initialize the unified configuration
config = Config()

####################################
#  HELPER FUNCTIONS
####################################

def load_model_and_tokenizer(config):
    """
    Load a model and tokenizer from a specified path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

####################################
#  MAIN LOGIC
####################################

def main():
    # Initialize wandb
    wandb.init(project="kto")

    # Load models and tokenizer
    print("Loading models and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    ref_model, _ = load_model_and_tokenizer(config)
    print("Models and tokenizer loaded.")

    # Load and process datasets using the specified function
    print("Processing dataset...")
    dataset = config.process_dataset_func()
    print("Dataset processed.")

    # Initialize trainer
    print("Initializing trainer...")
    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=KTOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            warmup_ratio=config.warmup_ratio,
            bf16=config.bf16,
            logging_first_step=config.logging_first_step,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config({
            "use_peft": config.use_peft,
            "lora_target_modules": config.lora_target_modules,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        }),
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
    wandb.log(metrics)

    # Save model and optionally push to hub
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()

    print("Process completed.")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
