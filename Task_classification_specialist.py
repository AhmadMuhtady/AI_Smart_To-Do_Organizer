import os
import json
import time
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError, BadRequestError

load_dotenv(override=True)
groq_ai_key = os.getenv("GROQAI_API_KEY")
groq_ai = OpenAI(api_key=groq_ai_key, base_url="https://api.groq.com/openai/v1")

today = date.today()
formatted_date = today.strftime("%d-%m-%Y")
weekday_name = today.strftime("%A")

SYSTEM_PROMPT_MODEL_2 = f"""You are Model 2: Task Normalization & Classification Specialist.
Your input is a raw JSON payload containing extracted tasks. Your job is to normalize and classify each item into a finalized structure.

Current Reference Date: {weekday_name}, {formatted_date}.

### Normalization & Classification Rules:

1. Strict 1-to-1 Mapping & Completeness (MANDATORY):
   - You MUST process every single input task in the exact order received.
   - The number of output tasks MUST EXACTLY EQUAL the number of input tasks (e.g., if given 20 tasks, you must return exactly 20 tasks).
   - NEVER skip, merge, deduplicate, summarize, truncate, or omit any tasks under any circumstance.

2. Title:
   - Format as a clean, concise, grammatically correct verb phrase.
   - Retain original intent without adding unstated facts.

3. Description (Preserve Useful Context & Consequences):
   - ALWAYS preserve meaningful context: purpose, reasons, locations, specific stakes, penalties, or sub-day timing details that do NOT fit into a calendar date (e.g., "from the store", "for an oil change", "for the client meeting", "for CEO meeting", "afternoon", "or they will shut off power in 10 minutes", "or receive a penalty").
   - If the extracted context contains useful information not already represented by the title or deadline, PRESERVE IT in `description`.
   - Remove ONLY details that merely duplicate what is already expressed by the title or the normalized calendar deadline (e.g., omit words like "today" or "tomorrow" if the deadline field captures them).
   - If there is genuinely zero extra context beyond the title and calendar deadline, output null.

4. Priority (Consistency Policy):
   - Priority reflects task importance and stakes, COMPLETELY INDEPENDENT of completion status (a completed tax return or client project retains its true priority, never Low simply because it is done).
   - Low: purely optional, discretionary, casual, or vague items easily postponed without real negative consequence ("organize garage sometime", "cancel subscription whenever").
   - Medium (Standard Responsibilities): normal, concrete responsibilities, routine tasks, assignments, essays, chores, or errands ("buy milk today", "pick up dry cleaning tomorrow", "call dentist", "finish history essay by Friday", "submit assignment next Friday", "presentation for regular meeting").
     * Firm Rule: A calendar deadline alone does NOT automatically elevate a normal responsibility or standard school/work deliverable to High. Standard coursework (assignments, essays) and standard work tasks remain Medium by default unless explicit high-stakes factors exist.
   - High: significant importance with explicit high stakes, executive/major-client visibility, or major examinations ("presentation for CEO meeting tomorrow", "study for final exam", "quarterly executive financial filing", "critical client deliverable").
   - Urgent: requires immediate action AND carries an immediate severe penalty/failure within hours or minutes ("pay bill right now or power cuts in 10 minutes", "pay parking ticket in 15 minutes or receive penalty"). A nearby standard deadline alone is NEVER Urgent.

5. Deadline & Calendar Verification:
   - Output format: DD-MM-YYYY or null.
   - "today" / immediate timing -> current date ({formatted_date}).
   - "tomorrow" -> next calendar day.
   - Weekday Resolution Rule:
     * Unqualified weekday (e.g., "Friday") or "this Friday" -> the nearest upcoming occurrence of that day.
     * "next Friday" -> the Friday of the following week (7 days after the nearest Friday).
     * MANDATORY INTERNAL CHECK: Before finalizing a date, confirm that the DD-MM-YYYY date mathematically falls on the requested weekday relative to {weekday_name} {formatted_date}.
   - Vague timing ("sometime", "later this month maybe") -> null.
   - Past completed history ("yesterday", "last week") -> null.

6. Category:
   - Select from: Work, Study, Personal, Shopping, Health, Finance, General.
   - Purpose beats shallow keywords (e.g., buying a textbook is Study, buying groceries is Shopping, dentist appointment is Health). Use General only when nothing else fits.

7. Status:
   - Base ONLY on `state_expression`.
   - Pending: default state.
   - In Progress: explicit phrase showing work has started ("started studying", "working on", "started cleaning").
   - Completed: explicit phrase showing work is already finished ("already finished", "done", "finished filing").
"""

Model_2_Classification_Schema = {
    "type": "json_schema",
    "name": "Smart_To_Do_Organizer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "A list of normalized, classified tasks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Concise verb phrase."},
                        "description": {"type": ["string", "null"], "description": "Meaningful consequences, stakes, extra context, or sub-day timing."},
                        "priority": {"type": "string", "enum": ["Low", "Medium", "High", "Urgent"]},
                        "deadline": {"type": ["string", "null"], "description": "DD-MM-YYYY format, or null."},
                        "category": {"type": "string", "enum": ["Work", "Study", "Personal", "Shopping", "Health", "Finance", "General"]},
                        "status": {"type": "string", "enum": ["Pending", "In Progress", "Completed"]},
                    },
                    "required": ["title", "description", "priority", "deadline", "category", "status"],
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
        ui_message = "Authentication issue encountered."
    elif isinstance(error, NotFoundError):
        error_type = "model_not_found"
        ui_message = "Model not found."
    elif isinstance(error, json.JSONDecodeError):
        error_type = "invalid_json"
        ui_message = "Invalid JSON produced by model."
    elif isinstance(error, APIError):
        error_type = "api_error"
        ui_message = "API service error."
    else:
        error_type = "unknown_error"
        ui_message = "Unexpected error."

    return {"error": error_type, "message": ui_message, "detail": str(error)}

def _call_model_2_api(task_list: list) -> dict:
    """Issues a single classified batch with explicit task count reinforcement in prompt."""
    input_payload_json = json.dumps({"tasks": task_list}, indent=2)
    user_prompt = f"""Normalize and classify the following {len(task_list)} extracted tasks.
You must return exactly {len(task_list)} tasks in the exact same sequence.

INPUT DATA:
{input_payload_json}"""

    response = groq_ai.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MODEL_2},
            {"role": "user", "content": user_prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Smart_To_Do_Organizer",
                "strict": True,
                "schema": Model_2_Classification_Schema["schema"]
            }
        },
        temperature=0.1,
        max_tokens=4096
    )
    return json.loads(response.choices[0].message.content)

def _is_validation_failure(err: Exception) -> bool:
    err_str = str(err).lower()
    return "json_validate_failed" in err_str or "schema" in err_str or "400" in err_str

def task_classification_specialist(model_1_output: dict) -> dict:
    if not model_1_output or "tasks" not in model_1_output:
        return {"error": "invalid_input", "message": "Model 1 did not return valid task data."}

    input_tasks = model_1_output["tasks"]
    expected_count = len(input_tasks)


    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            result = _call_model_2_api(input_tasks)
            output_tasks = result.get("tasks", [])
            
            # External Programmatic Validation: Check if the model dropped tasks
            if len(output_tasks) == expected_count:
                return result
            
            print(f"[Model 2 Count Warning] Incomplete return on attempt {attempt + 1}: Expected {expected_count}, got {len(output_tasks)}.")
            if attempt < max_retries:
                print("Retrying full batch in 1.5s...")
                time.sleep(1.5)
                continue
            else:
                print("Max full-batch retries reached. Activating 10+10 batch fallback...")
                break

        except Exception as e:
            if _is_validation_failure(e) and attempt < max_retries:
                print(f"[Model 2 Warning] Transient validation failure on attempt {attempt + 1}. Retrying in 1.5s...")
                time.sleep(1.5)
                continue
            elif attempt == max_retries:
                print("[Model 2 Warning] Full-batch calls failed. Activating 10+10 batch fallback...")
                break
            else:
                return handle_ai_error(e)


    try:
        chunk_size = 10
        merged_tasks = []
        for i in range(0, expected_count, chunk_size):
            chunk = input_tasks[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            print(f"[Model 2 Fallback] Processing batch chunk {chunk_num} ({len(chunk)} tasks)...")
            
            # Call batch chunk
            chunk_result = _call_model_2_api(chunk)
            chunk_output_tasks = chunk_result.get("tasks", [])
            

            if len(chunk_output_tasks) != len(chunk):
                print(f"[Model 2 Fallback] Chunk {chunk_num} dropped tasks (expected {len(chunk)}, got {len(chunk_output_tasks)}). Retrying chunk once...")
                time.sleep(1.5)
                chunk_result = _call_model_2_api(chunk)
                chunk_output_tasks = chunk_result.get("tasks", [])
            
            merged_tasks.extend(chunk_output_tasks)

        return {"tasks": merged_tasks}
    except Exception as e:
        return handle_ai_error(e)