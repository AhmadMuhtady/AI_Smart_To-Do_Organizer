import os
import json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError

load_dotenv(override=True)
groq_ai_key = os.getenv("GROQAI_API_KEY")
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

3. Revisions and Clarifications in the Same Sentence:
   - If the same action and object are mentioned more than once in the same sentence, do NOT create multiple tasks. Treat later wording as a clarification or revision of the same task unless a genuinely different action is introduced.

4. Corrected or Revised Timing:
   - If the user revises or updates timing with words such as "actually", "instead", "rather", or similar correction language, keep the task once and use the latest timing expression as `time_expression`.

5. Action Verb Belongs in Task Text (Never a Bare Noun):
   - `task_text` MUST include the core action verb together with its direct object/modifiers (e.g., "work on the slide deck", "finish history essay", "cancel that subscription", "pay electricity bill").
   - A task cannot be a standalone noun phrase like "the slide deck".

6. State Expression Boundaries:
   - `state_expression` captures ONLY the auxiliary language expressing intent, modal obligation, progress, or completion (e.g., "need to", "I've started", "should probably", "already finished", "remind me to").
   - It must NEVER swallow the main action verb (e.g., in "I've started working on the deck", the state is "I've started" and the task is "work on the slide deck"). If no explicit state/progress wording is present, output null.

7. Timing Belongs Exclusively in Time Expression:
   - ANY phrase describing when or how soon a task should happen—including vague, non-specific, or loose timing (e.g., "later this month maybe", "whenever I get a chance", "sometime soon", "right now", "tomorrow")—MUST go into `time_expression`.
   - Never leak temporal or scheduling phrases into `context` or `state_expression`.

8. Task Context Boundaries:
   - `context` is strictly for non-temporal auxiliary facts: purpose, reasons, stakes, locations, or consequences (e.g., "for an oil change", "from the store", "for the client meeting").
   - Discard throwaway filler words (e.g., "well whatever") or set `context` to null if there are no real extra stakes or reasons.

9. Task Isolation & Completeness:
   - Context, time, and state must originate ONLY from that specific task.
   - Never nest an actionable sub-task inside another task's context.
   - Before finishing, do a completeness check to confirm all actionable items in the prompt are extracted exactly once.

### Field Definitions:
- `task_text`: The full actionable core (verb + direct object + essential modifiers).
- `context`: Auxiliary reasons, purposes, locations, or consequences. Null if none.
- `time_expression`: The EXACT timing phrase provided (including vague or relative timing). Null if none.
- `state_expression`: The EXACT auxiliary phrase indicating intent, progress, or completion status (excluding the action verb). Null if none.

Do not invent missing information. If a detail is missing in the user's text, output null for that field."""

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
                            "description": "The core actionable task, containing the action verb and direct object.",
                        },
                        "context": {
                            "type": ["string", "null"],
                            "description": "Auxiliary reasons, stakes, purpose, or consequences. Excludes timing.",
                        },
                        "time_expression": {
                            "type": ["string", "null"],
                            "description": "The EXACT time phrase (specific or vague). Excludes intent/state.",
                        },
                        "state_expression": {
                            "type": ["string", "null"],
                            "description": "The EXACT auxiliary phrase indicating intent, progress, or completion.",
                        },
                    },
                    "required": ["task_text", "context", "time_expression", "state_expression"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}

def handle_ai_error(error: Exception) -> dict:
    if isinstance(error, RateLimitError):
        error_type = "rate_limit"
        ui_message = "Rate limit reached. Please wait a few seconds and try again."
    elif isinstance(error, AuthenticationError):
        error_type = "authentication_error"
        ui_message = "Authentication issue encountered. Please verify your API key."
    elif isinstance(error, NotFoundError):
        error_type = "model_not_found"
        ui_message = "The requested AI model is unavailable."
    elif isinstance(error, json.JSONDecodeError):
        error_type = "invalid_json"
        ui_message = "The model generated a malformed response."
    elif isinstance(error, APIError):
        error_type = "api_error"
        ui_message = "The AI service experienced an error."
    else:
        error_type = "unknown_error"
        ui_message = "Something unexpected occurred."

    return {"error": error_type, "message": ui_message, "detail": str(error)}

def task_extraction_specialist(clean_text: str):
    user_prompt = f"""Extract all actionable tasks from the following text according to your system instructions:

TEXT START
{clean_text}
TEXT END"""
    try:
        response = groq_ai.responses.create(
            model="qwen/qwen3.8-27b",
            instructions=SYSTEM_PROMPT_MODEL_1,
            input=user_prompt,
            text={"format": Model_1_Extraction_Schema},
        )
        return json.loads(response.output_text)
    except Exception as e:
        return handle_ai_error(e)