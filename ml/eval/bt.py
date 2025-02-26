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
    old_generations_file: str
    new_generations_file: str   
    output_file: str = 'bt_results.json'


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


def bradley_terry_comparison(old_rewards, new_rewards):
    """
    Perform Bradley-Terry comparison between two sets of model generations.

    Args:
        old_rewards (list): List of dictionaries for the OLD model's generations and rewards.
        new_rewards (list): List of dictionaries for the NEW model's generations and rewards.

    Returns:
        list: Comparison results including preferred outputs and probabilities.
        dict: Metrics summary including percentage preferred and average probabilities.
    """
    results = []
    new_preferred_count = 0
    old_preferred_count = 0
    probabilities = []

    for ix in range(len(old_rewards)):
        old = sft_rewards[ix]
        new = kto_rewards[ix]

        # Ensure prompts match
        assert old['prompt'] == new['prompt'], f"ERROR: Prompts at index {ix} do not match."

        # Compute Bradley-Terry probability
        new_reward = torch.tensor(kto['reward'], dtype=torch.float32)
        old_reward = torch.tensor(sft['reward'], dtype=torch.float32)
        prob_new_preferred = torch.sigmoid(kto_reward - old_reward).item()

        probabilities.append(prob_new_preferred)
        preferred_model = 'new' if prob_kto_preferred > 0.5 else 'old'

        # Count preferences
        if preferred_model == 'new':
            new_preferred_count += 1
        else:
            old_preferred_count += 1

        # Log results
        bt_result = {
            'prompt': old['prompt'],
            'old_output': sft['output'],
            'new_output': kto['output'],
            'old_reward': sft['reward'],
            'new_reward': kto['reward'],
            'preferred': preferred_model,
            'prob_new_preferred': prob_kto_preferred
        }
        results.append(bt_result)

    # Calculate metrics
    total_examples = len(old_rewards)
    metrics = {
        'total_examples': total_examples,
        'new_preferred_percentage': 100 * kto_preferred_count / total_examples,
        'old_preferred_percentage': 100 * sft_preferred_count / total_examples,
        'avg_probability_new_preferred': sum(probabilities) / total_examples
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
    print(f"Percentage preferred - KTO model: {metrics['new_preferred_percentage']:.2f}%")
    print(f"Percentage preferred - SFT model: {metrics['old_preferred_percentage']:.2f}%")
    print(f"Average probability of KTO model being preferred: {metrics['avg_probability_new_preferred']:.4f}")


####################################
# MAIN SCRIPT
####################################

def main():
    # Initialize script arguments
    args = ScriptArguments()

    # Load data
    print("Loading data...")
    old_rewards = load_rewards(args.sft_generations_file)
    new_rewards = load_rewards(args.kto_generations_file)

    # Perform Bradley-Terry comparison
    print("Performing Bradley-Terry comparison...")
    results, metrics = bradley_terry_comparison(old_rewards, new_rewards)

    # Save results
    save_results(results, args.output_file)

    # Print metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()



