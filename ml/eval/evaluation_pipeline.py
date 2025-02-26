###########
# IMPORTS #
###########
from reward_eval import process_evaluation
from generate import generate_files
from alpaca import alpaca_evaluator
from bt import bradley_terry_comparison, save_results, print_metrics

##################
# M-REWARD BENCH #
##################



#############
# EVALUATOR #
#############
'''
Evaluation Pipeline 

Parameters: 
eval_dataset: list of dictionaries that contain the prompt and response in the same form as below: 
            [{"prompt": "How are you?", "output": "I'm doing great!"}, {"prompt": "What's your name?", "output": "Assistant"}]
reward_output_filepath: string (must end in .json) that represents the path of the output of the reward score evaluation
model: base model that is being evaluated (defaults to starter base model - Aya-23-8B )

'''
def evaluator_master_fn(eval_dataset: list[dict], 
                        reward_output_filepath: str, 
                        model="CohereForAI/aya-23-8B"):
    
    # 1. Reward score evaluation: 
    args = {
        'bfloat16': False,  
        'reward_output_fmt': '1-0',
        'apply_sigmoid_to_reward': False,
        'per_device_batch_size': 8,
        'output_filepath': reward_output_filepath + '.json',
        'result_filename': None,
    }
    process_evaluation(args, model_name=model, eval_data_list_dict=eval_dataset)

    # 2. 

