import json
import torch
from dataclasses import dataclass

####################################
# SCRIPT ARGUMENTS
####################################

@dataclass
class ScriptArguments:
    """
    Arguments for the Bradley-Terry evaluation script.
    """
    sft_generations_file: str = '/raid/lingo/jen_ben/HF-RLHF/eval/test/gen_examples_idan_mini.json'
    kto_generations_file: str = '/raid/lingo/jen_ben/HF-RLHF/eval/test/gen_examples_idan_mini.json'
    output_file: str = 'bt_results_test_mini.json'


####################################
# FUNCTIONS
####################################

def load_rewards(file_path):
    """
    Load the rewards from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing model generations and rewards.

    Returns:
        list: List of dictionaries with prompts, outputs, and rewards.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def bradley_terry_comparison(sft_rewards, kto_rewards):
    """
    Perform Bradley-Terry comparison between two sets of model generations.

    Args:
        sft_rewards (list): List of dictionaries for the SFT model's generations and rewards.
        kto_rewards (list): List of dictionaries for the KTO model's generations and rewards.

    Returns:
        list: Comparison results including preferred outputs and probabilities.
        dict: Metrics summary including percentage preferred and average probabilities.
    """
    results = []
    kto_preferred_count = 0
    sft_preferred_count = 0
    probabilities = []

    for ix in range(len(sft_rewards)):
        sft = sft_rewards[ix]
        kto = kto_rewards[ix]

        # Ensure prompts match
        assert sft['prompt'] == kto['prompt'], f"ERROR: Prompts at index {ix} do not match."

        # Compute Bradley-Terry probability
        kto_reward = torch.tensor(kto['reward'], dtype=torch.float32)
        sft_reward = torch.tensor(sft['reward'], dtype=torch.float32)
        prob_kto_preferred = torch.sigmoid(kto_reward - sft_reward).item()

        probabilities.append(prob_kto_preferred)
        preferred_model = 'kto' if prob_kto_preferred > 0.5 else 'sft'

        # Count preferences
        if preferred_model == 'kto':
            kto_preferred_count += 1
        else:
            sft_preferred_count += 1

        # Log results
        bt_result = {
            'prompt': sft['prompt'],
            'sft_output': sft['output'],
            'kto_output': kto['output'],
            'sft_reward': sft['reward'],
            'kto_reward': kto['reward'],
            'preferred': preferred_model,
            'prob_kto_preferred': prob_kto_preferred
        }
        results.append(bt_result)

    # Calculate metrics
    total_examples = len(sft_rewards)
    metrics = {
        'total_examples': total_examples,
        'kto_preferred_percentage': 100 * kto_preferred_count / total_examples,
        'sft_preferred_percentage': 100 * sft_preferred_count / total_examples,
        'avg_probability_kto_preferred': sum(probabilities) / total_examples
    }

    return results, metrics


def save_results(results, output_path):
    """
    Save the comparison results to a JSON file.

    Args:
        results (list): List of comparison results.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")


def print_metrics(metrics):
    """
    Print evaluation metrics.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print("\nEVALUATION METRICS:")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Percentage preferred - KTO model: {metrics['kto_preferred_percentage']:.2f}%")
    print(f"Percentage preferred - SFT model: {metrics['sft_preferred_percentage']:.2f}%")
    print(f"Average probability of KTO model being preferred: {metrics['avg_probability_kto_preferred']:.4f}")


####################################
# MAIN SCRIPT
####################################

def main():
    # Initialize script arguments
    args = ScriptArguments()

    # Load data
    print("Loading data...")
    sft_rewards = load_rewards(args.sft_generations_file)
    kto_rewards = load_rewards(args.kto_generations_file)

    # Perform Bradley-Terry comparison
    print("Performing Bradley-Terry comparison...")
    results, metrics = bradley_terry_comparison(sft_rewards, kto_rewards)

    # Save results
    save_results(results, args.output_file)

    # Print metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()



# import json
# import torch

# output_file_path = 'bt_results.json'
# ref_generations_rewards_file_path = 'ref_models_generations_reward_trl-libqwen1.5-1.8b-sft.json'
# finetuned_generations_rewards_file_path = 'finetuned_models_generations_reward_trl-libqwen1.5-1.8b-sft.json'

# # Open and read JSON files
# with open(ref_generations_rewards_file_path, 'r') as f:
#     ref_rewards = json.load(f)

# with open(finetuned_generations_rewards_file_path, 'r') as g:
#     finetuned_rewards = json.load(g)

# # assert len(ref_rewards) != len(finetuned_rewards), 'ERROR: files are not with the same length.'

# results = []
# finetuned_preffered = 0
# for ix in range(len(ref_rewards)):
#     ref = ref_rewards[ix]
#     finetuned = finetuned_rewards[ix]
#     assert ref['prompt'] == finetuned['prompt'], 'ERROR: ref and finetuned prompt are not the same.'

#     # Bradely Terry
#     finetuned_reward = torch.tensor(finetuned['reward'], dtype=torch.float32)
#     ref_reward = torch.tensor(ref['reward'], dtype=torch.float32)
#     prob_finetuned_preferred = torch.sigmoid(finetuned_reward - ref_reward)


#     if prob_finetuned_preferred > 0.5:
#         finetuned_preffered +=1
#         print(f'example {ix}: finetuned preffered')
#     else:
#         print(f'example {ix}: ref preffered')

#     # log results
#     bt_result = {}
#     bt_result['prompt'] = ref['prompt']
#     bt_result['ref_output'] = ref['output']
#     bt_result['finetuned_output'] = finetuned['output']
#     bt_result['ref_reward'] = ref['output']
#     bt_result['finetuned_reward'] = finetuned['output']
#     bt_result['preffered'] = 'finetuned' if prob_finetuned_preferred > 0.5 else 'ref'
#     results.append(bt_result)


# # save results in json files

# with open(output_file_path, "w") as f:
#     json.dump(results, f, indent=4)

# print('BT EVALUATION COMPLETED.')
