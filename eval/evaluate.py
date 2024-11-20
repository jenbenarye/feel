import sys
import os
from typing import Any, Dict, List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format
from tqdm import tqdm

# Add script directory to system path for importing local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from eval.utils import jload, jdump
from eval.evaluate_arguments import EvalArguments


# set `device` to "cuda" if a GPU is available. otherwise, defaults to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model():
    # loads a specified reward model and sets it to use the GPU ("cuda")
    # CHANGE FUNCTION DEPENDING OF THE MODEL YOU LOAD
    model = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", num_labels=1).to("cuda")
    return model


def create_tokenizer():
    # loads the tokenizer that pairs with the model for encoding the text data
    tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", use_auth_token=True)
    return tokenizer


def MyAccelerator(mixed_precision):
    # wrap `Accelerator` to set up model handling with mixed-precision (to save memory)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    return accelerator


#####################################
# Idan's script from here
#####################################


def main():

    # Parse evaluation arguments from `EvalArguments`
    parser = transformers.HfArgumentParser((EvalArguments, ))
    args, = parser.parse_args_into_dataclasses()

    # set `mixed_precision` based on `args.bfloat16` (if true use bf16, otherwise fp16)
    mixed_precision = 'bf16' if args.bfloat16 else 'fp16'
    args.mixed_precision = mixed_precision

    # initialize `MyAccelerator` with the chosen mixed precision setting
    accelerator = MyAccelerator(
        mixed_precision=mixed_precision,
    )


    # load model and tokenizer
    model = create_model()
    if 't5' not in args.model_name_or_path:
        # t5 models where trained with fp32
        model = accelerator.prepare(model)
    model.eval()

    tokenizer = create_tokenizer()

    print("Output file path:", args.output_filepath)

    # load LM generations data from `args.output_filepath` + handles cases where it’s a single file or directory.
    filenames = []
    eval_data_list_dict = []
    if os.path.isfile(args.output_filepath):
        print(f'Loading data from {args.output_filepath}...')
        eval_data_list_dict.append(jload(args.output_filepath))
        filenames.append(args.output_filepath)
    elif os.path.isdir(args.output_filepath):
        print(f'Loading data from {args.output_filepath}...')
        for filename in os.listdir(args.output_filepath):
            if filename.endswith('.json'):
                print(f'Loaded file {filename}')
                eval_data_list_dict.append(jload(os.path.join(args.output_filepath, filename)))
                filenames.append(os.path.join(args.output_filepath, filename))
    else:
        raise Exception('Output file(s) not found!')


    # process each file and call `evaluate_data()` to calculate reward scores
    for filename, eval_data_dict in zip(filenames, eval_data_list_dict):
        eval_data = evaluate_data(args, model, tokenizer, eval_data_dict)

        if args.result_filename is None:
            path_to_result = os.path.basename(filename).split('.json')[0] + f"_reward_{args.model_name_or_path.replace('/', '')}.json"
        else:
            path_to_result = args.result_filename

        print(f'Saving results to file {path_to_result}...')
        jdump(eval_data, path_to_result)


def get_reward_output_fn(reward_output_fmt: str, apply_sigmoid_to_reward: bool):
    # defines the reward output function format based on `reward_output_fmt`
    if reward_output_fmt is None:
        reward_output_fn = lambda x: x.squeeze().cpu().detach().numpy().tolist()
    elif reward_output_fmt == '0':
        reward_output_fn = lambda x: x.squeeze().cpu().detach().softmax(dim=-1).numpy()[0].tolist()
    elif reward_output_fmt == '1':
        reward_output_fn = lambda x: x.squeeze().cpu().detach().softmax(dim=-1).numpy()[1].tolist()
    elif reward_output_fmt == '1-0':
        reward_output_fn = lambda x: (x.squeeze().cpu().detach().softmax(dim=-1).numpy()[1] - x.squeeze().cpu().detach().softmax(dim=-1).numpy()[0]).tolist()
    else:
        raise NotImplementedError(f'Unsupported reward output format: {reward_output_fmt}')

    # Apply sigmoid transformation if `apply_sigmoid_to_reward` is true
    if apply_sigmoid_to_reward:
        reward_output_fn = lambda x: torch.sigmoid(torch.tensor(x)).numpy().tolist()

    return reward_output_fn


@torch.inference_mode()
def evaluate_data(args: EvalArguments, model, tokenizer, eval_data_list_dict) -> List[Dict[str, Any]]:
    """Given a generated dataset, evaluate it using the reward model

    args: argparse.Namespace, the arguments to use
    reward_model: reward_model_module.RewardModel, the reward model to use
    eval_data_list_dict: List[Dict[str, Any]], the generated data to evaluate
    """

    pbar = tqdm(total=len(eval_data_list_dict), desc="eval")
    rewards_list = []
    reward_output_fn = get_reward_output_fn(args.reward_output_fmt, args.apply_sigmoid_to_reward)

    print('Evaluating reward scores...')

    # Split `eval_data_list_dict` into batches for processing
    for idx in range(0, len(eval_data_list_dict), args.per_device_batch_size):
        if len(eval_data_list_dict) > (idx + args.per_device_batch_size):
            batch_list_dict = eval_data_list_dict[idx:idx+args.per_device_batch_size]
        else:
            batch_list_dict = eval_data_list_dict[idx:]

        # create formatted text from prompts and outputs for tokenization
        if 'prompt' in batch_list_dict[0]:
            batch_full_outputs = [l['prompt'] + ' ' + l['output'] for l in batch_list_dict]
        else:
            print('Overriding with custom prompt format')
            prompt_fmt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: {output}"
            for l in batch_list_dict:
                l['output'] = l['output'].split('.')[0] + '.'
            batch_full_outputs = [prompt_fmt.format_map(l) for l in batch_list_dict]

        # tokenize and send the batched text to the model’s device
        encoded_full_responses = tokenizer(batch_full_outputs, return_tensors="pt", padding=True, truncation=True)
        encoded_full_responses = encoded_full_responses.to(model.device) # i added this

        # generate reward scores and stores them in `rewards_list`
        reward_outputs = model(**encoded_full_responses)
        rewards = reward_output_fn(reward_outputs.logits)
        rewards_list.extend(rewards if isinstance(rewards, list) else [rewards])

        # update progress bar after each batch is processed
        pbar.update(len(batch_list_dict))

    print('Combining reward outputs into outputs...')

    # add calculated rewards to each item in `eval_data_list_dict`
    for j in range(len(eval_data_list_dict)):
        eval_data_list_dict[j]['reward'] = rewards_list[j]
        eval_data_list_dict[j]['reward_model'] = args.model_name_or_path + args.model_pretrained_lora_weights if args.model_pretrained_lora_weights is not None else args.model_name_or_path

    print('Finished evaluating reward scores!')

    print('Mean reward score: ', sum(rewards_list) / len(rewards_list))
    print('Std reward score: ', torch.tensor(rewards_list).std().item())

    return eval_data_list_dict


if __name__ == '__main__':
    main()
