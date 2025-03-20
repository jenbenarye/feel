import os
import random
import uuid
from base64 import b64encode
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Optional
import json

import spaces
import spaces
import gradio as gr
from feedback import save_feedback, scheduler
from gradio.components.chatbot import Option
from huggingface_hub import InferenceClient
from pandas import DataFrame
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


BASE_MODEL = os.getenv("MODEL", "CohereForAI/aya-expanse-8b")
ZERO_GPU = (
    bool(os.getenv("ZERO_GPU", False)) or True
    if str(os.getenv("ZERO_GPU")).lower() == "true"
    else False
)
TEXT_ONLY = (
    bool(os.getenv("TEXT_ONLY", False)) or True
    if str(os.getenv("TEXT_ONLY")).lower() == "true"
    else False
)


def create_inference_client(
    model: Optional[str] = None, base_url: Optional[str] = None
) -> InferenceClient:
    """Create an InferenceClient instance with the given model or environment settings.
    This function will run the model locally if ZERO_GPU is set to True.
    This function will run the model locally if ZERO_GPU is set to True.

    Args:
        model: Optional model identifier to use. If not provided, will use environment settings.

    Returns:
        InferenceClient: Configured client instance
    """
    if ZERO_GPU:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True)
        return {
            "pipeline": pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2000,
            ),
            "tokenizer": tokenizer
        }
    else:
        return InferenceClient(
            token=os.getenv("HF_TOKEN"),
            model=model if model else (BASE_MODEL if not base_url else None),
            base_url=base_url,
        )


CLIENT = create_inference_client()


def load_languages() -> dict[str, str]:
    """Load languages from JSON file or persistent storage"""
    # First check if we have persistent storage available
    persistent_path = Path("/data/languages.json")
    local_path = Path(__file__).parent / "languages.json"
    
    # Check if persistent storage is available and writable
    use_persistent = False
    if Path("/data").exists() and Path("/data").is_dir():
        try:
            # Test if we can write to the directory
            test_file = Path("/data/write_test.tmp")
            test_file.touch()
            test_file.unlink()  # Remove the test file
            use_persistent = True
        except (PermissionError, OSError):
            print("Persistent storage exists but is not writable, falling back to local storage")
            use_persistent = False
    
    # Use persistent storage if available and writable, otherwise fall back to local file
    if use_persistent and persistent_path.exists():
        languages_path = persistent_path
    else:
        languages_path = local_path
        
        # If persistent storage is available and writable but file doesn't exist yet,
        # copy the local file to persistent storage
        if use_persistent:
            try:
                # Ensure local file exists first
                if local_path.exists():
                    import shutil
                    # Copy the file to persistent storage
                    shutil.copy(local_path, persistent_path)
                    languages_path = persistent_path
                    print(f"Copied languages to persistent storage at {persistent_path}")
                else:
                    # Create an empty languages file in persistent storage
                    with open(persistent_path, "w", encoding="utf-8") as f:
                        json.dump({"English": "You are a helpful assistant."}, f, ensure_ascii=False, indent=2)
                    languages_path = persistent_path
                    print(f"Created new languages file in persistent storage at {persistent_path}")
            except Exception as e:
                print(f"Error setting up persistent storage: {e}")
                languages_path = local_path  # Fall back to local path if any error occurs
    
    with open(languages_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Initial load
LANGUAGES = load_languages()

# User agreement text
USER_AGREEMENT = """
You have been asked to participate in a research study conducted by Lingo Lab from the Computer Science and Artificial Intelligence Laboratory at the Massachusetts Institute of Technology (M.I.T.), together with huggingface. 

The purpose of this study is the collection of multilingual human feedback to improve language models. As part of this study you will interat with a language model in a langugage of your choice, and provide indication to wether its reponses are helpful or not.

Your name and personal data will never be recorded. You may decline further participation, at any time, without adverse consequences.There are no foreseeable risks or discomforts for participating in this study. Note participating in the study may pose risks that are currently unforeseeable. If you have questions or concerns about the study, you can contact the researchers at leshem@mit.edu. If you have any questions about your rights as a participant in this research (E-6610), feel you have been harmed, or wish to discuss other study-related concerns with someone who is not part of the research team, you can contact the M.I.T. Committee on the Use of Humans as Experimental Subjects (COUHES) by phone at (617) 253-8420, or by email at couhes@mit.edu.

Clicking on the next button at the bottom of this page indicates that you are at least 18 years of age and willingly agree to participate in the research voluntarily.
"""


def add_user_message(history, message):
    if isinstance(message, dict) and "files" in message:
        for x in message["files"]:
            history.append({"role": "user", "content": {"path": x}})
        if message["text"] is not None:
            history.append({"role": "user", "content": message["text"]})
    else:
        history.append({"role": "user", "content": message})
    return history, gr.Textbox(value=None, interactive=False)


def format_system_message(language: str, history: list):
    system_message = [
        {
            "role": "system",
            "content": LANGUAGES.get(language, LANGUAGES["English"]),
        }
    ]
    if history and history[0]["role"] == "system":
        history = history[1:]
    history = system_message + history
    return history


def format_history_as_messages(history: list):
    messages = []
    current_role = None
    current_message_content = []

    if TEXT_ONLY:
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        return messages

    if TEXT_ONLY:
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        return messages

    for entry in history:
        content = entry["content"]

        if entry["role"] != current_role:
            if current_role is not None:
                messages.append(
                    {"role": current_role, "content": current_message_content}
                )
            current_role = entry["role"]
            current_message_content = []

        if isinstance(content, tuple):  # Handle file paths
            for temp_path in content:
                if space_host := os.getenv("SPACE_HOST"):
                    url = f"https://{space_host}/gradio_api/file%3D{temp_path}"
                else:
                    url = _convert_path_to_data_uri(temp_path)
                current_message_content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
        elif isinstance(content, str):  # Handle text
            current_message_content.append({"type": "text", "text": content})

    if current_role is not None:
        messages.append({"role": current_role, "content": current_message_content})

    return messages


def _convert_path_to_data_uri(path) -> str:
    mime_type, _ = guess_type(path)
    with open(path, "rb") as image_file:
        data = image_file.read()
        data_uri = f"data:{mime_type};base64," + b64encode(data).decode("utf-8")
    return data_uri


def _is_file_safe(path) -> bool:
    try:
        return Path(path).is_file()
    except Exception:
        return ""


def _process_content(content) -> str | list[str]:
    if isinstance(content, str) and _is_file_safe(content):
        return _convert_path_to_data_uri(content)
    elif isinstance(content, list) or isinstance(content, tuple):
        return _convert_path_to_data_uri(content[0])
    return content


def _process_rating(rating) -> int:
    if isinstance(rating, str):
        return 0
    elif isinstance(rating, int):
        return rating
    else:
        raise ValueError(f"Invalid rating: {rating}")


def add_fake_like_data(
    history: list,
    conversation_id: str,
    session_id: str,
    language: str,
    liked: bool = False,
) -> None:
    data = {
        "index": len(history) - 1,
        "value": history[-1],
        "liked": liked,
    }
    _, dataframe = wrangle_like_data(
        gr.LikeData(target=None, data=data), history.copy()
    )
    submit_conversation(
        dataframe=dataframe,
        conversation_id=conversation_id,
        session_id=session_id,
        language=language,
    )


@spaces.GPU
def call_pipeline(messages: list, language: str):
    if ZERO_GPU:
        # Format the messages using the tokenizer's chat template
        tokenizer = CLIENT["tokenizer"]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
        )
        
        # Call the pipeline with the formatted text
        response = CLIENT["pipeline"](
            formatted_prompt,
            clean_up_tokenization_spaces=False,
            max_length=2000,
            return_full_text=False,
        )
        
        # Extract the generated content
        content = response[0]["generated_text"]
        return content
    else:
        response = CLIENT(
            messages,
            clean_up_tokenization_spaces=False,
            max_length=2000,
        )
        content = response[0]["generated_text"][-1]["content"]
        return content


def respond(
    history: list,
    language: str,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> list:  # -> list:
    """Respond to the user message with a system message

    Return the history with the new message"""
    messages = format_history_as_messages(history)
    if ZERO_GPU:
        content = call_pipeline(messages, language)
    else:
        response = CLIENT.chat.completions.create(
            messages=messages,
            max_tokens=2000,
            stream=False,
            seed=seed,
            temperature=temperature,
        )
        content = response.choices[0].message.content
    if ZERO_GPU:
        content = call_pipeline(messages, language)
    else:
        response = CLIENT.chat.completions.create(
            messages=messages,
            max_tokens=2000,
            stream=False,
            seed=seed,
            temperature=temperature,
        )
        content = response.choices[0].message.content
    message = gr.ChatMessage(role="assistant", content=content)
    history.append(message)
    return history


def update_dataframe(dataframe: DataFrame, history: list) -> DataFrame:
    """Update the dataframe with the new message"""
    data = {
        "index": 9999,
        "value": None,
        "liked": False,
    }
    _, dataframe = wrangle_like_data(
        gr.LikeData(target=None, data=data), history.copy()
    )
    return dataframe


def wrangle_like_data(x: gr.LikeData, history) -> DataFrame:
    """Wrangle conversations and liked data into a DataFrame"""

    if isinstance(x.index, int):
        liked_index = x.index
    else:
        liked_index = x.index[0]

    output_data = []
    for idx, message in enumerate(history):
        if isinstance(message, gr.ChatMessage):
            message = message.__dict__
        if idx == liked_index:
            if x.liked is True:
                message["metadata"] = {"title": "liked"}
            elif x.liked is False:
                message["metadata"] = {"title": "disliked"}

        if not isinstance(message["metadata"], dict):
            message["metadata"] = message["metadata"].__dict__
        rating = message["metadata"].get("title")
        if rating == "liked":
            message["rating"] = 1
        elif rating == "disliked":
            message["rating"] = -1
        else:
            message["rating"] = 0

        message["chosen"] = ""
        message["rejected"] = ""
        if message["options"]:
            for option in message["options"]:
                if not isinstance(option, dict):
                    option = option.__dict__
                message[option["label"]] = option["value"]
        else:
            if message["rating"] == 1:
                message["chosen"] = message["content"]
            elif message["rating"] == -1:
                message["rejected"] = message["content"]

        output_data.append(
            dict(
                [(k, v) for k, v in message.items() if k not in ["metadata", "options"]]
            )
        )

    return history, DataFrame(data=output_data)


def wrangle_edit_data(
    x: gr.EditData,
    history: list,
    dataframe: DataFrame,
    conversation_id: str,
    session_id: str,
    language: str,
) -> list:
    """Edit the conversation and add negative feedback if assistant message is edited, otherwise regenerate the message

    Return the history with the new message"""
    if isinstance(x.index, int):
        index = x.index
    else:
        index = x.index[0]

    original_message = gr.ChatMessage(
        role="assistant", content=dataframe.iloc[index]["content"]
    ).__dict__

    if history[index]["role"] == "user":
        # Add feedback on original and corrected message
        add_fake_like_data(
            history=history[: index + 2],
            conversation_id=conversation_id,
            session_id=session_id,
            language=language,
            liked=True,
        )
        add_fake_like_data(
            history=history[: index + 1] + [original_message],
            conversation_id=conversation_id,
            session_id=session_id,
            language=language,
        )
        history = respond(
            history=history[: index + 1],
            language=language,
            temperature=random.randint(1, 100) / 100,
            seed=random.randint(0, 1000000),
        )
        return history
    else:
        # Add feedback on original and corrected message
        add_fake_like_data(
            history=history[: index + 1],
            conversation_id=conversation_id,
            session_id=session_id,
            language=language,
            liked=True,
        )
        add_fake_like_data(
            history=history[:index] + [original_message],
            conversation_id=conversation_id,
            session_id=session_id,
            language=language,
        )
        history = history[: index + 1]
        # add chosen and rejected options
        history[-1]["options"] = [
            Option(label="chosen", value=x.value),
            Option(label="rejected", value=original_message["content"]),
        ]
        return history


def wrangle_retry_data(
    x: gr.RetryData,
    history: list,
    dataframe: DataFrame,
    conversation_id: str,
    session_id: str,
    language: str,
) -> list:
    """Respond to the user message with a system message and add negative feedback on the original message

    Return the history with the new message"""
    add_fake_like_data(
        history=history,
        conversation_id=conversation_id,
        session_id=session_id,
        language=language,
    )

    # Return the history without a new message
    history = respond(
        history=history[:-1],
        language=language,
        temperature=random.randint(1, 100) / 100,
        seed=random.randint(0, 1000000),
    )
    return history, update_dataframe(dataframe, history)


def submit_conversation(dataframe, conversation_id, session_id, language):
    """ "Submit the conversation to dataset repo"""
    if dataframe.empty or len(dataframe) < 2:
        gr.Info("No feedback to submit.")
        return (gr.Dataframe(value=None, interactive=False), [])

    dataframe["content"] = dataframe["content"].apply(_process_content)
    dataframe["rating"] = dataframe["rating"].apply(_process_rating)
    conversation = dataframe.to_dict(orient="records")
    conversation_data = {
        "conversation": conversation,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "conversation_id": conversation_id,
        "language": language,
    }
    save_feedback(input_object=conversation_data)
    return (gr.Dataframe(value=None, interactive=False), [])


def open_add_language_modal():
    return gr.Group(visible=True)

def close_add_language_modal():
    return gr.Group(visible=False)

def save_new_language(lang_name, system_prompt):
    """Save the new language and system prompt to persistent storage if available, otherwise to local file."""
    # First determine where to save the file
    persistent_path = Path("/data/languages.json")
    local_path = Path(__file__).parent / "languages.json"
    
    # Check if persistent storage is available and writable
    use_persistent = False
    if Path("/data").exists() and Path("/data").is_dir():
        try:
            # Test if we can write to the directory
            test_file = Path("/data/write_test.tmp")
            test_file.touch()
            test_file.unlink()  # Remove the test file
            use_persistent = True
        except (PermissionError, OSError):
            print("Persistent storage exists but is not writable, falling back to local storage")
            use_persistent = False
    
    # Use persistent storage if available and writable, otherwise fall back to local file
    languages_path = persistent_path if use_persistent else local_path
    
    # Load existing languages
    if languages_path.exists():
        with open(languages_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    
    # Add the new language to JSON
    data[lang_name] = system_prompt

    # Save the updated languages
    with open(languages_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # If we're using persistent storage, also update the local file as backup
    if use_persistent and local_path != persistent_path:
        try:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error updating local backup: {e}")
    
    # Update the dropdown choices
    new_choices = list(LANGUAGES.keys())
    
    # Return a message that will trigger a JavaScript refresh
    return gr.Group(visible=False), gr.HTML("<script>window.location.reload();</script>"), gr.Dropdown(choices=new_choices)


css = """
.options.svelte-pcaovb {
    display: none !important;
}
.option.svelte-pcaovb {
    display: none !important;
}
.retry-btn {
    display: none !important;
}
/* Style for the add language button */
button#add-language-btn {
    padding: 0 !important;
    font-size: 30px !important;
    font-weight: bold !important;
}
/* Style for the user agreement container */
.user-agreement-container {
    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
}
"""
# /* Style for the user agreement container */
# .user-agreement-container {
#     background-color: white !important;
#     box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
# }
# /* Ensure the markdown inside the container inherits the background */
# .user-agreement-container > div {
#     background-color: white !important;
# }
# /* Target all elements inside the container */
# .user-agreement-container * {
#     background-color: white !important;
# }
# """

with gr.Blocks(css=css) as demo:
    # State variable to track if user has consented
    user_consented = gr.State(False)
    
    # Landing page with user agreement
    with gr.Group(visible=True) as landing_page:
        gr.Markdown("# Welcome to FeeL")
        with gr.Group(elem_classes=["user-agreement-container"]):
            gr.Markdown(USER_AGREEMENT)
        consent_btn = gr.Button("I agree")
    
    # Main application interface (initially hidden)
    with gr.Group(visible=False) as main_app:
        ##############################
        # Chatbot
        ##############################
        gr.Markdown("""
        # ♾️ FeeL - a real-time Feedback Loop for LMs
        """)

        with gr.Accordion("About") as explanation:
            gr.Markdown(f"""
            FeeL is a collaboration between Hugging Face and MIT.
            It is a community-driven project to provide a real-time feedback loop for VLMs, where your feedback is continuously used to fine-tune the underlying models.
            The [dataset](https://huggingface.co/datasets/{scheduler.repo_id}), [code](https://github.com/huggingface/feel) and [models](https://huggingface.co/collections/feel-fl/feel-models-67a9b6ef0fdd554315e295e8) are public.

            Start by selecting your language, chat with the model with text and images and provide feedback in different ways.

            - ✏️ Edit a message
            - 👍/👎 Like or dislike a message
            - 🔄 Regenerate a message

        Feedback is automatically submitted allowing you to continue chatting, but you can also submit and reset the conversation by clicking "💾 Submit conversation" (under the chat) or trash the conversation by clicking "🗑️" (upper right corner).
        """)
        language = gr.Dropdown(
            choices=list(LANGUAGES.keys()), label="Language", interactive=True
        )

    with gr.Blocks(css="""
    #add-language-btn {
        background: url('os.path.abspath("app/feel-add-icon.png")') no-repeat center;
        background-size: contain;
        width: 50px;
        height: 50px;
        border: none;
        cursor: pointer;
    }
""") as demo:
        add_button = gr.Button("", elem_id="add-language-btn")
        output = gr.Textbox(label="Status")

        session_id = gr.Textbox(
            interactive=False,
            value=str(uuid.uuid4()),
            visible=False,
        )

        conversation_id = gr.Textbox(
            interactive=False,
            value=str(uuid.uuid4()),
            visible=False,
        )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            editable="all",
            bubble_full_width=False,
            value=[
                {
                    "role": "system",
                    "content": LANGUAGES[language.value],
                }
            ],
            type="messages",
            feedback_options=["Like", "Dislike"],
        )

        chat_input = gr.Textbox(
            interactive=True,
            placeholder="Enter message or upload file...",
            show_label=False,
            submit_btn=True,
        )

        with gr.Accordion("Collected feedback", open=False):
            dataframe = gr.Dataframe(wrap=True, label="Collected feedback")

        submit_btn = gr.Button(value="💾 Submit conversation", visible=False)

    # Function to show main app after consent
    def show_main_app():
        return gr.Group(visible=False), gr.Group(visible=True), True

    # Connect consent button to show main app
    consent_btn.click(
        fn=show_main_app,
        inputs=[],
        outputs=[landing_page, main_app, user_consented]
    )

    ##############################
    # Deal with feedback
    ##############################

    language.change(
        fn=format_system_message,
        inputs=[language, chatbot],
        outputs=[chatbot],
    )

    chat_input.submit(
        fn=add_user_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
    ).then(respond, inputs=[chatbot, language], outputs=[chatbot]).then(
        lambda: gr.Textbox(interactive=True), None, [chat_input]
    ).then(update_dataframe, inputs=[dataframe, chatbot], outputs=[dataframe]).then(
        submit_conversation,
        inputs=[dataframe, conversation_id, session_id, language],
    )

    chatbot.like(
        fn=wrangle_like_data,
        inputs=[chatbot],
        outputs=[chatbot, dataframe],
        like_user_message=False,
    ).then(
        submit_conversation,
        inputs=[dataframe, conversation_id, session_id, language],
    )

    chatbot.retry(
        fn=wrangle_retry_data,
        inputs=[chatbot, dataframe, conversation_id, session_id, language],
        outputs=[chatbot, dataframe],
    )

    chatbot.edit(
        fn=wrangle_edit_data,
        inputs=[chatbot, dataframe, conversation_id, session_id, language],
        outputs=[chatbot],
    ).then(update_dataframe, inputs=[dataframe, chatbot], outputs=[dataframe])

    gr.on(
        triggers=[submit_btn.click, chatbot.clear],
        fn=submit_conversation,
        inputs=[dataframe, conversation_id, session_id, language],
        outputs=[dataframe, chatbot],
    ).then(
        fn=lambda x: str(uuid.uuid4()),
        inputs=[conversation_id],
        outputs=[conversation_id],
    )

    def on_app_load():
        return str(uuid.uuid4())

    demo.load(
        fn=on_app_load,
        inputs=None,
        outputs=session_id
    )

    add_language_btn.click(
        fn=lambda: gr.Group(visible=True),
        outputs=[add_language_modal]
    )

    cancel_language_btn.click(
        fn=lambda: gr.Group(visible=False),
        outputs=[add_language_modal]
    )

    save_language_btn.click(
        fn=save_new_language,
        inputs=[new_lang_name, new_system_prompt],
        outputs=[add_language_modal, refresh_html, language]
    )

demo.launch()
