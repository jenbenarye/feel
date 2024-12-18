import json
from typing import TYPE_CHECKING, List, Literal, Union

from datasets import Dataset, concatenate_datasets
from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import CombineOutputs, GeneratorStep, KeepColumns, Step, StepInput
from distilabel.steps.tasks import TextGeneration
from typing_extensions import override

CHOSEN_TEMPLATE = """
You are provide with a conversation between a human and an AI assistant. 
The final message is of poort quality positively. Your task is to regenerate one of high quality.
{% for message in conversation %}
{{ message["role"] }}: {{ message["content"] }}
{% endfor %}
Replacement improved message:
""".rstrip()

CHOSEN_SYSTEM_PROMPT = "You are a helpful AI assistant. Your task is to regenerate high quality responses to user queries, when other assistants go wrong."

REJECT_TEMPLATE = """
You are provide with a conversation between a human and an AI assistant.
The final message has been rated positively. Your task is to regenerate a POOR QUALITYresponse.
{% for message in conversation %}
{{ message["role"] }}: {{ message["content"] }}
{% endfor %}
Replacement improved message:
""".rstrip()

REJECT_SYSTEM_PROMPT = "You are a helpful AI assistant. Your task is to regenerate high quality responses to user queries, when other assistants go wrong."


class FilterConversationRatings(Step):
    """Filters conversations based on the rating of the last message."""

    target_column: Union[Literal["chosen"], Literal["rejected"]]
    batch_size: int = 5

    @override
    def process(self, dataset: StepInput) -> "GeneratorStepOutput":

        column_rating_map = {
            "chosen": 1,
            "rejected": -1,
        }

        target_rating = column_rating_map[self.target_column]

        for batch_start in range(0, len(dataset), self.batch_size):
            batch = dataset[batch_start : batch_start + self.batch_size]
            filtered_batch = []
            for conversation in batch:
                for row in batch:
                    _conversation = row["conversation"]
                    conversation = None
                    for idx, message in enumerate(_conversation, 1):
                        if not isinstance(message["rating"], int):
                            continue
                        if message["rating"] == target_rating:
                            conversation = _conversation[:idx]
                            break
                    if conversation:
                        filtered_batch.append({"conversation": conversation})
            yield filtered_batch

    @property
    def outputs(self) -> "StepColumns":
        return ["conversation"]


class AppendToConversationStep(Step):
    """Appends a generated message to a conversation."""

    @property
    def inputs(self) -> "StepColumns":
        return ["generation", "conversation"]

    @property
    def outputs(self) -> "StepColumns":
        return ["generated_conversation", "conversation"]

    def process(self, inputs: StepInput) -> "StepOutput":

        for input in inputs:
            if not input["generation"]:
                continue
            if not input["conversation"]:
                continue
            input["generated_conversation"] = [
                {"role": message["role"], "content": message["content"]}
                for message in input["conversation"][:-1]
            ] + [{"role": "assistant", "content": input["generation"]}]
            input["conversation"] = [
                {"role": message["role"], "content": message["content"]}
                for message in input["conversation"]
            ]
        yield inputs


with Pipeline(
    name="conversation_rejection",
    description="Generate a chosen response to a rejected conversation.",
) as rejection_pipeline:

    rejected_dataset = FilterConversationRatings(target_column="rejected")

    chosen_text_gen = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        system_prompt=CHOSEN_SYSTEM_PROMPT,
        template=CHOSEN_TEMPLATE,
        columns=["conversation"],
    )

    append_chosen = AppendToConversationStep(
        output_mappings={
            "generated_conversation": "chosen",
            "conversation": "rejected",
        },
    )

    keep_columns = KeepColumns(
        columns=["chosen", "rejected"],
    )

    rejected_dataset >> chosen_text_gen >> append_chosen >> keep_columns

with Pipeline(
    name="conversation_chosen",
    description="Generate a rejected response to a chosen conversation.",
) as chosen_pipeline:

    chosen_dataset = FilterConversationRatings(target_column="chosen")

    rejected_text_gen = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        system_prompt=REJECT_SYSTEM_PROMPT,
        template=REJECT_TEMPLATE,
        columns=["conversation"],
    )
    append_rejected = AppendToConversationStep(
        output_mappings={
            "generated_conversation": "rejected",
            "conversation": "chosen",
        },
    )
    keep_columns = KeepColumns(
        columns=["chosen", "rejected"],
    )
    chosen_dataset >> rejected_text_gen >> append_rejected >> keep_columns

if __name__ == "__main__":

    dataset_path = "example_data.json"
    data = json.load(open(dataset_path))

    dataset = Dataset.from_list(data)
    rejected_dataset = rejection_pipeline.run(dataset=dataset, use_cache=False)
    chosen_dataset = chosen_pipeline.run(dataset=dataset, use_cache=False)

    dataset = concatenate_datasets(
        dsets=[rejected_dataset["default"]["train"], chosen_dataset["default"]["train"]]
    )
