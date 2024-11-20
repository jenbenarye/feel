import torch
from dataclasses import dataclass

from accelerate import PartialState
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str = "trl-lib/kto-mix-14k"


# Initialize the arguments directly
script_args = ScriptArguments(
    dataset_name="trl-lib/kto-mix-14k"
)

training_args = KTOConfig(
    output_dir="/raid/lingo/jen_ben/HF-RLHF/kto_nov_2", # MODFIFY
    num_train_epochs=100,
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=8,
    logging_steps=10,
    eval_steps=500,
    warmup_ratio=0.1,
    bf16=True,
    logging_first_step=True
)

model_args = ModelConfig(
    model_name_or_path="trl-lib/qwen1.5-1.8b-sft",
    # any additional model-specific arguments
)

# Load a pretrained model
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
)
print(f'loaded model')

# load a tokenaizer
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# If we are aligning a base model, we use ChatML as the default template
if tokenizer.chat_template is None:
    model, tokenizer = setup_chat_format(model, tokenizer)
print(f'loaded tokenizer')

# Load the dataset
dataset = load_dataset(script_args.dataset_name)

# If needed, reformat a DPO-formatted dataset (prompt, chosen, rejected) to a KTO-format (prompt, completion, label)
dataset = maybe_unpair_preference_dataset(dataset, num_proc=training_args.dataset_num_proc)
print(f'loaded dataset')


# Apply chat template
def format_dataset(example):
    example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
    example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
    return example


# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
with PartialState().local_main_process_first():
    dataset = dataset.map(format_dataset, num_proc=training_args.dataset_num_proc)



# Initialize the KTO trainer
trainer = KTOTrainer(
    model,
    ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=get_peft_config(model_args),
)

print(f'start training')

trainer.train()

print(f'finished training')

metrics = trainer.evaluate()
print(f'metrics: \n {metrics}')
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# Save and push to hub
trainer.save_model(training_args.output_dir)
if training_args.push_to_hub:
    trainer.push_to_hub(dataset_name=script_args.dataset_name)
