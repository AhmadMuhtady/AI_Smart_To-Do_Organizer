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

2. Split Multiple Independent Tasks:
   - If a single sentence contains multiple distinct actionable tasks that do NOT contradict or replace each other, split them and return each as a separate task object.

3. Explicit Replacement / Superseding Handling (CRITICAL - Single Final Task Only):
   - When the user clearly replaces or corrects an earlier instruction using superseding language such as:
     * "actually"
     * "instead"
     * "actually ... instead"
     * "no, do ... instead"
     * "rather"
     * "change that to"
   - You MUST extract ONLY the final intended task that supersedes the earlier one. Do NOT emit two tasks.
   - The earlier canceled/replaced action and its timing are discarded.
     * Example: "remind me to email Sarah the updated client proposal by Friday afternoon, but actually make sure to send her the draft slide deck tomorrow before lunch instead"
       -> Output EXACTLY ONE task:
          task_text: "send Sarah the draft slide deck" (or "send her the draft slide deck")
          time_expression: "tomorrow before lunch"
          state_expression: "make sure to" (or "remind me to")
          context: null
     * Example: "remind me to cancel that subscription whenever I get a chance, but actually do it later this month maybe, well whatever."
       -> Output EXACTLY ONE task:
          task_text: "cancel that subscription"
          time_expression: "later this month maybe"
          state_expression: "remind me to"
          context: null
   - Firm Guardrail: Do NOT globally merge separate, unrelated tasks. Only supersede when the phrasing explicitly indicates that the latter replaces the former.

4. Action Verb Belongs Exclusively in Task Text (Never a Bare Noun):
   - `task_text` MUST ALWAYS contain the actionable verb and its direct object/modifiers.
   - NEVER drop the action verb to leave a bare noun phrase.
     * BAD: `task_text`: "an important client report"
     * GOOD: `task_text`: "finish an important client report"
     * BAD: `task_text`: "the slide deck"
     * GOOD: `task_text`: "work on the slide deck"

5. State Expression Boundary (Auxiliary Words Only):
   - `state_expression` MUST contain ONLY auxiliary words that convey intent, reminders, progress, or completion.
   - It must NEVER consume or duplicate the primary action verb.
     * "remind me to cancel that subscription" -> `state_expression`: "remind me to", `task_text`: "cancel that subscription"
     * "Already finished an important client report" -> `state_expression`: "Already finished", `task_text`: "finish an important client report"
     * "I've started working on the slide deck" -> `state_expression`: "started", `task_text`: "work on the slide deck"
     * If no auxiliary intent/status words exist, `state_expression` must be null.

6. Timing Boundary (Past, Present, Future, and Vague):
   - `time_expression` captures ANY phrase indicating WHEN an action was performed, is performed, or should be performed.
   - Past timing phrases (e.g., "yesterday", "last week") belong strictly in `time_expression`, NEVER in `context`.
     * "I already finished filing my taxes yesterday" -> `task_text`: "file my taxes", `state_expression`: "already finished", `time_expression`: "yesterday", `context`: null.
   - Action timing must not include negative consequences or penalties (e.g., "in 10 minutes" inside "or power cuts in 10 minutes" is a consequence and stays in context).

7. Context Boundary (No Self-Duplication):
   - `context` is strictly for auxiliary reasons, locations, purposes, or consequences (e.g., "for an oil change", "from the store", "or they will shut off power in 10 minutes").
   - NEVER duplicate the task or parts of the task inside `context`. If there are no genuinely extra stakes or details, `context` must be null.

8. Task Isolation & Completeness:
   - Context, time, and state must originate ONLY from that specific task.
   - Never nest an actionable sub-task inside another task's context.

### Field Definitions:
- `task_text`: The full actionable core (verb + direct object + modifiers). Never a bare noun.
- `context`: Auxiliary reasons, purposes, locations, consequences, or penalties. Never duplicates task_text or timing. Null if none.
- `time_expression`: The EXACT timing phrase (past, relative, specific, or vague) for when the task happened or should happen. Null if none.
- `state_expression`: The EXACT auxiliary phrase indicating intent, reminders, progress, or completion. Excludes the action verb. Null if none.

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
                            "description": "Auxiliary reasons, stakes, purpose, consequences, or penalties. Excludes action timing.",
                        },
                        "time_expression": {
                            "type": ["string", "null"],
                            "description": "The EXACT time phrase for when the task occurred or should occur.",
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
        response = groq_ai.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_MODEL_1},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Task_Extraction",
                    "strict": True,
                    "schema": Model_1_Extraction_Schema["schema"]
                }
            },
            temperature=0.0,
            max_tokens=4096
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return handle_ai_error(e)