import os
import random
import uuid
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from base64 import b64encode
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Optional
import json
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

import spaces
import spaces
import gradio as gr
from feedback import save_feedback, scheduler
from gradio.components.chatbot import Option
from huggingface_hub import InferenceClient
from pandas import DataFrame
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


BASE_MODEL = os.getenv("MODEL", "google/gemma-3-12b-pt")
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
) -> InferenceClient | dict:
    """Create an InferenceClient instance with the given model or environment settings.
    This function will run the model locally if ZERO_GPU is set to True.
    This function will run the model locally if ZERO_GPU is set to True.

    Args:
        model: Optional model identifier to use. If not provided, will use environment settings.
        base_url: Optional base URL for the inference API.

    Returns:
        Either an InferenceClient instance or a dictionary with pipeline and tokenizer
    """
    if ZERO_GPU:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=False)
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


def get_persistent_storage_path(filename: str) -> tuple[Path, bool]:
    """Check if persistent storage is available and return the appropriate path.

    Args:
        filename: The name of the file to check/create

    Returns:
        A tuple containing (file_path, is_persistent)
    """
    persistent_path = Path("/data") / filename
    local_path = Path(__file__).parent / filename

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

    return (persistent_path if use_persistent else local_path, use_persistent)


def load_languages() -> dict[str, str]:
    """Load languages from JSON file or persistent storage"""
    languages_path, use_persistent = get_persistent_storage_path("languages.json")
    local_path = Path(__file__).parent / "languages.json"

    # If persistent storage is available but file doesn't exist yet, copy the local file to persistent storage
    if use_persistent and not languages_path.exists():
        try:
            if local_path.exists():
                import shutil
                shutil.copy(local_path, languages_path)
                print(f"Copied languages to persistent storage at {languages_path}")
            else:
                with open(languages_path, "w", encoding="utf-8") as f:
                    json.dump({"English": "You are a helpful assistant."}, f, ensure_ascii=False, indent=2)
                print(f"Created new languages file in persistent storage at {languages_path}")
        except Exception as e:
            print(f"Error setting up persistent storage: {e}")
            languages_path = local_path  # Fall back to local path if any error occurs

    if not languages_path.exists() and local_path.exists():
        languages_path = local_path

    if languages_path.exists():
        with open(languages_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        default_languages = {"English": "You are a helpful assistant."}
        return default_languages

LANGUAGES = load_languages()

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


def format_system_message(language: str):
    system_message = [
        {
            "role": "system",
            "content": LANGUAGES.get(language, LANGUAGES["English"]),
        },
        {
            "role": "user",
            "content": f"Start by asking me a question in {language}."
        }
    ]
    response = call_pipeline(system_message)
    new_system_message = [
        {
            "role": "system",
            "content": LANGUAGES.get(language, LANGUAGES["English"]),
        },
        {
            "role": "assistant",
            "content": response
        }
    ]
    return new_system_message


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
def call_pipeline(messages: list):
    """Call the appropriate model pipeline based on configuration"""
    if ZERO_GPU:
        tokenizer = CLIENT["tokenizer"]
        # Ensure messages follow the proper alternating pattern
        formatted_messages = []
        prev_role = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip empty messages
            if not content.strip():
                continue

            # Enforce alternating pattern
            if role == prev_role:
                # If same role repeats, combine with previous message or skip
                continue

            # Only allow "user" and "assistant" roles
            if role not in ["user", "assistant"]:
                # Convert to proper role or skip
                continue

            formatted_messages.append(msg)
            prev_role = role

        # Ensure we start with user message
        if formatted_messages and formatted_messages[0]["role"] != "user":
            formatted_messages = formatted_messages[1:]

        # Now use the properly formatted messages
        formatted_prompt = tokenizer.apply_chat_template(
            formatted_messages,  # Use the fixed messages
            tokenize=False,
            add_generation_prompt=True
        )

        response = CLIENT["pipeline"](
            formatted_prompt,
            clean_up_tokenization_spaces=False,
            max_length=2000,
            return_full_text=False,
        )

        return response[0]["generated_text"]
    else:
        response = CLIENT(
            messages,
            clean_up_tokenization_spaces=False,
            max_length=2000,
        )
        return response[0]["generated_text"][-1]["content"]


def respond(
    history: list,
    language: str,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> list:
    """Respond to the user message with a system message

    Return the history with the new message"""
    messages = format_history_as_messages(history)

    if ZERO_GPU:
        content = call_pipeline(messages)
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

        if message["metadata"] is None:
            message["metadata"] = {}
        elif not isinstance(message["metadata"], dict):
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
    global LANGUAGES

    languages_path, use_persistent = get_persistent_storage_path("languages.json")
    local_path = Path(__file__).parent / "languages.json"

    if languages_path.exists():
        with open(languages_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[lang_name] = system_prompt

    with open(languages_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if use_persistent and local_path != languages_path:
        try:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error updating local backup: {e}")

    LANGUAGES.update({lang_name: system_prompt})
    return gr.Group(visible=False), gr.HTML("<script>window.location.reload();</script>"), gr.Dropdown(choices=list(LANGUAGES.keys()))


def save_contributor_email(email, name=""):
    """Save contributor email to persistent storage and send notification to admins"""
    print(f"[DEBUG] Starting save_contributor_email for: {email}, {name}")

    # Still save to persistent storage for record keeping
    emails_path, use_persistent = get_persistent_storage_path("contributors.json")
    print(f"[DEBUG] Using path: {emails_path}, persistent: {use_persistent}")

    # Read existing emails
    contributors = []
    try:
        if emails_path.exists():
            with open(emails_path, "r", encoding="utf-8") as f:
                contributors = json.load(f)
                print(f"[DEBUG] Loaded {len(contributors)} existing contributors")
        else:
            print(f"[DEBUG] No existing contributors file found at {emails_path}")
    except Exception as e:
        print(f"[DEBUG] Error reading contributors file: {e}")

    # Add new email with timestamp
    contributor_data = {
        "email": email,
        "name": name,
        "timestamp": datetime.now().isoformat()
    }
    contributors.append(contributor_data)
    print(f"[DEBUG] Added new contributor data: {contributor_data}")

    # Save back to file
    try:
        with open(emails_path, "w", encoding="utf-8") as f:
            json.dump(contributors, f, ensure_ascii=False, indent=2)
            print(f"[DEBUG] Successfully saved contributors file with {len(contributors)} entries")
    except Exception as e:
        print(f"[DEBUG] Error saving contributors file: {e}")

    # Send email notification to admins
    print(f"[DEBUG] Attempting to send notification email")
    try:
        send_notification_email(contributor_data)
        print(f"[DEBUG] Successfully sent notification email")
        return True
    except Exception as e:
        print(f"[DEBUG] Failed to send notification email: {e}")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        if hasattr(e, 'args'):
            print(f"[DEBUG] Error args: {e.args}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return False

def send_notification_email(contributor_data):
    """Send email notification to admins about new contributor using SendGrid API"""
    # Get configuration from environment variables
    sender_email = os.getenv("NOTIFICATION_EMAIL", "feel.notifications@gmail.com")
    recipient_email = os.getenv("ADMIN_EMAIL", "jen_ben@mit.edu")
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY", "")

    print(f"[DEBUG] Email configuration:")
    print(f"[DEBUG] - Sender Email: {sender_email}")
    print(f"[DEBUG] - Recipient Email: {recipient_email}")
    print(f"[DEBUG] - API Key Set: {'Yes' if sendgrid_api_key else 'No'}")

    # If no API key is set, log instead of sending
    if not sendgrid_api_key:
        print(f"[DEBUG] No SendGrid API key set, would send notification email about contributor: {contributor_data}")
        return False

    try:
        # Create message content
        html_content = f"""
        <html>
        <body>
            <h2>New FeeL Contributor Submission</h2>
            <p><strong>Name:</strong> {contributor_data.get('name', 'Not provided')}</p>
            <p><strong>Email:</strong> {contributor_data.get('email', 'Not provided')}</p>
            <p><strong>Timestamp:</strong> {contributor_data.get('timestamp', datetime.now().isoformat())}</p>
        </body>
        </html>
        """

        # Create mail message
        print(f"[DEBUG] Creating email message")
        message = Mail(
            from_email=sender_email,
            to_emails=recipient_email,
            subject='New FeeL Contributor Submission',
            html_content=html_content
        )

        # Send via API
        print(f"[DEBUG] Sending via SendGrid API")
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)

        print(f"[DEBUG] SendGrid API response code: {response.status_code}")

        # 202 is success for SendGrid
        if response.status_code == 202:
            print(f"[DEBUG] Email sent successfully via SendGrid API")
            return True
        else:
            print(f"[DEBUG] SendGrid API returned non-success status code: {response.status_code}")
            print(f"[DEBUG] Response body: {response.body}")
            return False

    except Exception as e:
        print(f"[DEBUG] Error in send_notification_email: {e}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return False

css = """
/* Style for the options and retry button */
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
    max-height: 300px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 5px;
    margin-bottom: 10px;
}
/* Style for the consent modal */
.consent-modal {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 9999 !important;
    background: var(--background-fill-primary) !important;
    padding: 20px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    max-width: 90% !important;
    width: 600px !important;
}
/* Overlay for the consent modal */
.modal-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background-color: rgba(0, 0, 0, 0.5) !important;
    z-index: 9998 !important;
}
.footer-banner {
    background-color: var(--background-fill-secondary);
    padding: 10px 20px;
    border-top: 1px solid var(--border-color-primary);
    margin-top: 20px;
    text-align: center;
}
.footer-banner p {
    margin: 0;
}
/* Language settings styling */
.language-settings-header {
    background-color: var(--primary-500);  /* Use Gradio's primary color */
    padding: 5px;
    border-radius: 8px 8px 0 0;
    margin-bottom: 0;
    color: var(--body-text-color);
    font-weight: bold;
}

.language-instruction {
    margin-top: 5px;
    margin-bottom: 5px;
    padding: 0 15px;
}

.language-container {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.language-dropdown {
    padding: 10px 15px 20px 15px;
}

.add-language-btn {
    background-color: var(--primary-500) !important;
    color: var(--body-text-color) !important;
    border: none !important;
    font-weight: bold !important;
    transition: background-color 0.3s !important;
}

.add-language-btn:hover {
    background-color: var(--primary-600) !important;
}

/* Yellow button styling - now using primary color variable */
button.yellow-btn {
    background-color: var(--primary-500) !important;
}

.footer-section {
    margin-top: 40px;
    border-top: 1px solid var(--border-color-primary);
    padding-top: 20px;
}
.admin-tools-accordion {
    max-width: 800px;
    margin: 0 auto;
}
.edit-instructions {
    padding: 10px 0;
    margin-top: 5px;
}
"""

def get_config(request: gr.Request):
    """Get configuration from cookies"""
    config = {
        "feel_consent": "false",
    }

    if request and request.cookies:
        for key in config.keys():
            if key in request.cookies:
                config[key] = request.cookies[key]

    return config["feel_consent"] == "true"

def initialize_consent_status(request: gr.Request):
    """Initialize consent status and language preference from cookies"""
    has_consent = get_config(request)
    return has_consent

js = '''function js(){
    window.set_cookie = function(key, value){
        document.cookie = key+'='+value+'; Path=/; SameSite=Strict';
        return [value];
    }
}'''



with gr.Blocks(css=css, js=js) as demo:
    # State variable to track if user has consented
    user_consented = gr.State(value=False)

    # Initialize language dropdown first
    language = gr.State(value="English")  # Default language state

    # Main application interface (initially hidden)
    with gr.Group() as main_app:
        with gr.Row():
            # Main content column (wider)
            with gr.Column(scale=3, elem_classes=["main-content"]):
                ##############################
                # Chatbot
                ##############################
                gr.Markdown("""
                # ♾️ FeeL: Improving LMs for All Languages
                """, elem_classes=["app-title"])

                with gr.Accordion("") as explanation:
                    gr.Markdown(f"""

                    **FeeL** (Feedback Loop) is a community-driven project by MIT, Hugging Face and IBM that aims to make language models better in *all languages*.

                    ### Why This Matters
                    Have you ever tried using an AI in your native language only to get responses that barely make sense? Most AI improvements prioritize widely spoken languages, while others fall behind. FeeL changes this by letting YOU shape how models respond in your language.

                    ### What You Can Do
                    1. **Select your language** from the dropdown menu (or add a new one if yours is missing)
                    2. **Chat with the model** in your language
                    3. **Provide feedback** on each response using:
                       - 👍/👎 Like or dislike responses
                       - ✏️ Edit responses to sound more natural or correct
                       - 🔄 Regenerate to try another response

                    Your feedback is directly used to fine-tune the model in real-time. The more you interact, the better the model becomes for your language community.

                    All [data](https://huggingface.co/datasets/{scheduler.repo_id}), [code](https://github.com/huggingface/feel) and [models](https://huggingface.co/collections/feel-fl/feel-models-67a9b6ef0fdd554315e295e8) are publicly available for research and development.

                    """)

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    editable="all",
                    value=format_system_message("English"),
                    type="messages",
                    feedback_options=["Like", "Dislike"],
                    height=600
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

            # Sidebar column (narrower)
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                with gr.Group(elem_classes=["language-container"]):
                    gr.Markdown("### Language Settings", elem_classes=["language-settings-header"])
                    gr.Markdown("Select your preferred language:", elem_classes=["language-instruction"])

                    with gr.Column(elem_classes=["language-dropdown"]):
                        language_dropdown = gr.Dropdown(
                            choices=list(load_languages().keys()),
                            value="English",
                            container=True,
                            show_label=False,
                        )

                    add_language_btn = gr.Button(
                        "Add New Language",
                        size="sm",
                        elem_classes=["add-language-btn"]
                    )

                # Create a hidden group instead of a modal
                with gr.Group(visible=False) as add_language_modal:
                    gr.Markdown("### Add New Language")
                    new_lang_name = gr.Textbox(label="Language Name", lines=1)
                    new_system_prompt = gr.Textbox(label="System Prompt", lines=4)
                    with gr.Row():
                        save_language_btn = gr.Button("Save")
                        cancel_language_btn = gr.Button("Cancel")

                # Add Contributors Tab
                with gr.Accordion("Thank You for Contributing", open=False):
                    gr.Markdown("""
                    We'd like to thank you for using FeeL and contributing to the improvement of multilingual language models.
                    If you'd like us to reach out to you, please leave your email below.

                    Your email will only be visible to the FeeL development team and won't be shared with others.
                    """)
                    contributor_email = gr.Textbox(
                        label="Your Email (optional)",
                        placeholder="email@example.com",
                        type="email"
                    )
                    contributor_name = gr.Textbox(
                        label="Your Name (optional)",
                        placeholder="Your name"
                    )
                    email_consent = gr.Checkbox(
                        label="I consent to being contacted by the FeeL team about my contributions",
                        value=False
                    )
                    submit_email_btn = gr.Button("Submit")
                    email_submit_status = gr.Markdown("")

        refresh_html = gr.HTML(visible=False)

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

    # Overlay for the consent modal
    with gr.Group(elem_classes=["modal-overlay"]) as consent_overlay:
        pass

    # Consent popup
    with gr.Group(elem_classes=["consent-modal"]) as consent_modal:
        gr.Markdown("# User Agreement")
        with gr.Group(elem_classes=["user-agreement-container"]):
            gr.Markdown(USER_AGREEMENT)
        consent_btn = gr.Button("I agree")

    # Add a contact footer at the bottom of the page
    with gr.Row(elem_classes=["footer-banner"]) as footer_banner:
        gr.Markdown("""
        ### Contact Us
        Have questions, requests, or ideas for how we can improve? Email us at: **jen_ben@mit.edu**
        """)

    # Add a subtle language management section at the bottom
    with gr.Row(elem_classes=["footer-section"]) as footer_section:
        with gr.Accordion("🔧 Admin Language Management", open=False, elem_classes=["admin-tools-accordion"]):
            # Removed the "Language File Manager" headline

            # Password authentication - button below password field
            admin_password = gr.Textbox(
                type="password",
                label="Admin Password",
                placeholder="Enter admin password"
            )
            auth_button = gr.Button("Authenticate", size="sm")

            auth_status = gr.Markdown("")

            # File management (initially hidden)
            with gr.Group(visible=False) as lang_editor_group:
                gr.Markdown("Edit the languages JSON file below:", elem_classes=["edit-instructions"])

                # Language file editor
                lang_json_editor = gr.Code(
                    language="json",
                    label="Languages JSON",
                    lines=15
                )

                with gr.Row():
                    load_button = gr.Button("Load Current Languages", size="sm")
                    save_button = gr.Button("Save Changes", size="sm", elem_classes=["yellow-btn"])

                result_message = gr.Markdown("")

    # Check consent on page load and show/hide components appropriately
    def initialize_consent_status():
        # This function will be called when the app loads
        return False  # Default to not consented

    def update_visibility(has_consent):
        # Show/hide components based on consent status
        return (
            gr.Group(visible=has_consent),  # main_app
            gr.Group(visible=not has_consent),  # consent_overlay
            gr.Group(visible=not has_consent),  # consent_modal
            gr.Group(visible=has_consent),  # footer_banner
            gr.Group(visible=has_consent)   # footer_section
        )

    # Initialize app with consent checking
    demo.load(
        fn=initialize_consent_status,
        outputs=user_consented,
        js=js
    ).then(
        fn=update_visibility,
        inputs=user_consented,
        outputs=[main_app, consent_overlay, consent_modal, footer_banner, footer_section]
    )

    # Update the consent button click handler
    consent_btn.click(
        fn=lambda: True,
        outputs=user_consented,
        js="() => set_cookie('feel_consent', 'true')"
    ).then(
        fn=update_visibility,
        inputs=user_consented,
        outputs=[main_app, consent_overlay, consent_modal, footer_banner, footer_section]
    )

    ##############################
    # Deal with feedback
    ##############################

    language_dropdown.change(
        fn=format_system_message,
        inputs=[language_dropdown],
        outputs=[chatbot],
    ).then(
        fn=lambda x: x,  # Update the language state
        inputs=[language_dropdown],
        outputs=[language]
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
        global LANGUAGES
        LANGUAGES = load_languages()
        language_choices = list(LANGUAGES.keys())
        default_language = language_choices[0] if language_choices else "English"

        return str(uuid.uuid4()), gr.Dropdown(choices=language_choices, value=default_language), default_language

    demo.load(
        fn=on_app_load,
        inputs=None,
        outputs=[session_id, language_dropdown, language]
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
        outputs=[add_language_modal, refresh_html, language_dropdown]
    )

    # Connect the events
    submit_email_btn.click(
        fn=lambda email, name, consent: "Thank you for your submission!" if consent else "Please provide consent to submit",
        inputs=[contributor_email, contributor_name, email_consent],
        outputs=[email_submit_status]
    ).then(
        fn=lambda email, name, consent: save_contributor_email(email, name) if consent else None,
        inputs=[contributor_email, contributor_name, email_consent],
        outputs=None
    )

    # Add the necessary functions
    def authenticate(password):
        """Authenticate the admin password"""
        correct_password = os.getenv("ADMIN_PASSWORD", "default_admin_password")
        if password == correct_password:
            return "✅ Authentication successful. You can now manage languages.", gr.Group(visible=True)
        else:
            return "❌ Incorrect password. Please try again.", gr.Group(visible=False)

    def load_languages_file():
        """Load the languages file from persistent storage"""
        languages_path, _ = get_persistent_storage_path("languages.json")
        try:
            with open(languages_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, "Languages file loaded successfully."
        except Exception as e:
            return "", f"Error loading languages file: {str(e)}"

    def save_languages_file(json_content):
        """Save the languages file to persistent storage"""
        try:
            # Validate JSON format
            languages_dict = json.loads(json_content)

            # Basic validation
            if not isinstance(languages_dict, dict):
                return "Error: Content must be a JSON object (dictionary)."

            for key, value in languages_dict.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    return f"Error: Keys and values must be strings. Issue with: {key}: {value}"

            # Save to file
            languages_path, _ = get_persistent_storage_path("languages.json")
            with open(languages_path, "w", encoding="utf-8") as f:
                f.write(json_content)

            return f"✅ Languages file updated successfully with {len(languages_dict)} languages."
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON format: {str(e)}"
        except Exception as e:
            return f"❌ Error saving languages file: {str(e)}"

    # Connect the event handlers
    auth_button.click(
        fn=authenticate,
        inputs=[admin_password],
        outputs=[auth_status, lang_editor_group]
    )

    load_button.click(
        fn=load_languages_file,
        inputs=[],
        outputs=[lang_json_editor, result_message]
    )

    save_button.click(
        fn=save_languages_file,
        inputs=[lang_json_editor],
        outputs=[result_message]
    )

demo.launch()
