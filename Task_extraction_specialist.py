import os
import json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError


load_dotenv(override=True)
groq_ai_key = os.getenv('GROQAI_API_KEY')
groq_ai = OpenAI(api_key=groq_ai_key, base_url="https://api.groq.com/openai/v1")


SYSTEM_PROMPT_MODEL_1 = """You are Model 1: Task Extraction Specialist. 
Your ONLY job is to extract actionable tasks from the user's text and map them to the provided JSON schema. 

You do NOT classify priority, you do NOT assign categories, you do NOT normalize dates, and you do NOT determine final task statuses. You are a pure extractor.

### Extraction Rules:
1. Extract EVERY Actionable Task:
   - Extract tasks regardless of their state. Include tasks that are pending, already started, already completed, vaguely scheduled, or extremely urgent.
   - Ignore non-actionable conversation (e.g., "The weather was amazing." results in no task).

2. Split Multiple Tasks:
   - If a single sentence contains multiple distinct actionable tasks, split them and return each as a separate task object.

3. Preserve Original Meaning & Phrasing:
   - Do not heavily rewrite or formalize the user's words. If they say "Take the car to the mechanic", keep it close to that. Do not invent corporate phrasing like "Schedule automotive maintenance".

### Field Definitions:
- `task_text`: The core action the user needs to do (or did).
- `context`: Any additional details, reasons, or stakes attached to the task. If none, use `null`.
- `time_expression`: Extract the EXACT timing phrase the user provided (e.g., "tomorrow", "Friday", "later this month maybe", "in 10 minutes", "yesterday"). Do NOT normalize this into a calendar date. If no time is mentioned, use `null`.
- `state_expression`: Extract the EXACT phrase indicating progress, intent, or completion (e.g., "already finished", "started working on", "need to", "remind me to"). If none, use `null`.

Do not invent missing information. If a detail is missing in the user's text, output `null` for that field."""


Model_1_Extraction_Schema = {
    "type": "json_schema",
    "name": "Task_Extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "A list of raw tasks extracted from the input text.",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_text": {
                            "type": "string",
                            "description": "The core actionable task, preserving the original meaning without heavy rewriting.",
                        },
                        "context": {
                            "type": ["string", "null"],
                            "description": "Additional details, stakes, or purpose associated with the task.",
                        },
                        "time_expression": {
                            "type": ["string", "null"],
                            "description": "The EXACT time phrase used by the user (e.g., 'tomorrow', 'Friday', 'in 10 minutes', 'yesterday'). Do not normalize to dates.",
                        },
                        "state_expression": {
                            "type": ["string", "null"],
                            "description": "The EXACT phrase indicating the state or progress (e.g., 'already finished', 'started working', 'need to').",
                        },
                    },
                    "required": [
                        "task_text", 
                        "context", 
                        "time_expression", 
                        "state_expression"
                    ],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}



raw_input_text = """
Buy milk from the store today.
I need to finish my history essay by Friday and also pick up my dry cleaning tomorrow.
Submit the quarterly report tomorrow afternoon.
Sometime soon I should probably organize my garage.
Take the car to the mechanic for an oil change.
Pay the electricity bill right now or they will shut off power in 10 minutes!
I already finished filing my taxes yesterday.
I've started working on the slide deck for the client meeting.
Man, what a crazy weekend. The weather was amazing.
Oh wait, remind me to cancel that subscription whenever I get a chance, but actually do it later this month maybe, well whatever.
"""


USER_PROMPT = f"""Extract all actionable tasks from the following text according to your system instructions:

TEXT START
{raw_input_text.strip()}
TEXT END"""


def handle_ai_error(error: Exception) -> dict:
    if isinstance(error, RateLimitError):
        error_type = "rate_limit"
        ui_message = "We're receiving a high volume of requests right now. Please wait a few seconds and try again."
    elif isinstance(error, AuthenticationError):
        error_type = "authentication_error"
        ui_message = "Authentication issue encountered. Please verify your API key or account settings."
    elif isinstance(error, NotFoundError):
        error_type = "model_not_found"
        ui_message = "The requested AI model is currently unavailable. Please check your configuration."
    elif isinstance(error, json.JSONDecodeError):
        error_type = "invalid_json"
        ui_message = "The model generated a malformed response. Please retry with your text."
    elif isinstance(error, APIError):
        error_type = "api_error"
        ui_message = "The AI service is experiencing a brief hiccup. Please try submitting again in a moment."
    else:
        error_type = "unknown_error"
        ui_message = "Something unexpected occurred. Please try again shortly."

    result = {"error": error_type, "message": ui_message, "detail": str(error)}
    print(f"Error: {result}")
    return result



def ask_extraction_specialist():
    try:
        response = groq_ai.responses.create(
            model = 'qwen/qwen3.8-27b',
            instructions = SYSTEM_PROMPT_MODEL_1,
            input = USER_PROMPT,
            text={'format': Model_1_Extraction_Schema}
        )

        results = json.loads(response.output_text)
        return results
    except json.JSONDecodeError as e:
        return handle_ai_error(e)
    except Exception as e:
        return handle_ai_error(e)
