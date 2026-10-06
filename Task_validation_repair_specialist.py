import os
import json
from datetime import date, timedelta
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError

load_dotenv(override=True)
groq_ai_key = os.getenv("GROQAI_API_KEY")
groq_ai = OpenAI(api_key=groq_ai_key, base_url="https://api.groq.com/openai/v1")

today = date.today()

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}

def resolve_expected_deadline(time_expr: str | None, ref_date: date = today) -> str | None:
    """Deterministically resolves relative or vague timing expressions to DD-MM-YYYY dates."""
    if not time_expr:
        return None

    raw = time_expr.strip().lower()

    # Vague or historical phrases -> null
    if any(k in raw for k in ["sometime", "later", "whenever", "yesterday", "last week", "past"]):
        return None

    # Immediate or same-day actions
    if "today" in raw or "right now" in raw or "in " in raw:
        return ref_date.strftime("%d-%m-%Y")

    # Next-day actions
    if "tomorrow" in raw:
        return (ref_date + timedelta(days=1)).strftime("%d-%m-%Y")

    # Calendar weekday calculation
    for name, target_idx in WEEKDAYS.items():
        if name in raw:
            current_idx = ref_date.weekday()
            days_ahead = (target_idx - current_idx) % 7
            if days_ahead == 0:
                days_ahead = 7

            if "next " + name in raw:
                days_ahead += 7

            return (ref_date + timedelta(days=days_ahead)).strftime("%d-%m-%Y")

    return None

SYSTEM_PROMPT_MODEL_3 = """You are Model 3: Final Field Validation & Repair Specialist.
Your input is an array of candidate tasks produced by Model 2 alongside extracted evidence from Model 1 and raw text.
Both arrays are strictly 1-to-1 aligned by index.

Your ONLY job is to audit individual fields and emit targeted field repairs for unambiguous errors.
Do NOT rewrite valid fields. Do NOT invent new descriptions. Do NOT add, delete, or reorder tasks.

### Strict Validation Rules:

1. Deadlines (Direct Match Against trusted_deadline):
   - You do NOT calculate dates. Python has pre-computed `trusted_deadline` for each task.
   - If candidate `deadline == trusted_deadline`: LEAVE IT ALONE.
   - If candidate `deadline != trusted_deadline`: Emit a repair setting `new_value: trusted_deadline`.

2. Description & Timing Policy:
   - Specific Sub-Day Timing (MANDATORY TO PRESERVE):
     Phrases specifying an exact time of day or interval (e.g., "afternoon", "in 15 minutes", "at 3 PM", "right now", "before noon") MUST be preserved in `description`.
     * Example: "tomorrow afternoon" -> deadline handles tomorrow, but "afternoon" must be in `description`.
     * Example: "in 15 minutes or receive a penalty" -> both the 15 minutes and penalty must be in `description`.
   - Vague Scheduling Language (DO NOT PUT IN DESCRIPTION):
     Vague timing phrases such as "sometime soon", "later this month maybe", "sometime next month", "whenever", or "sometime" DO NOT belong in `description`. If `description` is null for these, THAT IS CORRECT. Do NOT patch vague phrases into `description`.
   - Standard Calendar Deadlines:
     Words like "today", "tomorrow", "Friday", or "yesterday" captured by deadline must NOT be in `description`.
   - Context & Stakes:
     Genuine consequences, penalties, or stakes ("or they will shut off power in 10 minutes", "for an oil change") MUST be preserved in `description`.

3. Hallucination Removal:
   - Clear any details in `description` or `title` that have zero basis in Model 1 or raw text (set `new_value: null`).

4. Status & Progress Verification:
   - Compare `status` directly against Model 1's `state_expression`.
   - Completion evidence ("already finished", "done", "finished filing") -> `status: "Completed"`.
   - Progress evidence ("started", "in progress") -> `status: "In Progress"`.
   - Otherwise -> `status: "Pending"`.

5. Category Guardrail:
   - Only correct blatant blunders (e.g., studying for an exam marked as "Shopping" instead of "Study").
"""

Model_3_Validation_Schema = {
    "type": "json_schema",
    "name": "Pipeline_Field_Validation_And_Repair",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "passed": {
                "type": "boolean",
                "description": "True if zero defects are detected; False if any corrections are required."
            },
            "issues_found": {
                "type": "integer",
                "description": "Total count of field defects identified."
            },
            "corrections": {
                "type": "array",
                "description": "List of targeted field corrections.",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_index": {
                            "type": "integer",
                            "description": "Zero-based index of the task in the aligned candidate array."
                        },
                        "field": {
                            "type": "string",
                            "enum": ["title", "description", "priority", "deadline", "category", "status"],
                            "description": "Field being modified."
                        },
                        "old_value": {
                            "type": ["string", "null"],
                            "description": "Value before correction."
                        },
                        "new_value": {
                            "type": ["string", "null"],
                            "description": "Corrected replacement value."
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "wrong_deadline",
                                "wrong_status",
                                "wrong_category",
                                "wrong_priority",
                                "lost_context",
                                "hallucinated_information"
                            ],
                            "description": "Category of the defect."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Clear justification citing source text or Model 1 evidence."
                        }
                    },
                    "required": ["task_index", "field", "old_value", "new_value", "issue_type", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["passed", "issues_found", "corrections"],
        "additionalProperties": False
    }
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

def validate_and_repair_tasks(raw_text: str, model_1_output: dict, model_2_output: dict) -> dict:
    """Pre-calculates deterministic deadlines in Python and invokes Model 3 for field checks."""
    m1_tasks = model_1_output.get("tasks", [])
    m2_tasks = model_2_output.get("tasks", [])

    if len(m1_tasks) != len(m2_tasks):
        return {
            "error": "cardinality_mismatch",
            "message": f"Task count mismatch: Model 1 has {len(m1_tasks)} tasks, Model 2 has {len(m2_tasks)} tasks.",
            "detail": "Model 2 output must be retried or regenerated before validation."
        }

    annotated_m1 = []
    for item in m1_tasks:
        entry = dict(item)
        entry["trusted_deadline"] = resolve_expected_deadline(item.get("time_expression"))
        annotated_m1.append(entry)

    user_payload = {
        "raw_user_text": raw_text,
        "model_1_extractions_with_trusted_deadlines": annotated_m1,
        "model_2_candidates": m2_tasks
    }

    user_prompt = f"""Audit candidate tasks against source inputs and Python pre-computed trusted deadlines.
Only flag unambiguous defects:
- Deadlines that do NOT match trusted_deadline
- Missing specific sub-day timing that a date cannot represent (e.g. 'in 15 minutes', 'afternoon')
- Missing consequences, locations, or penalties
- Hallucinated details
- Incorrect status or category

Do NOT add vague timing phrases ('sometime soon', 'later this month maybe', 'sometime next month') into descriptions. Leaving description null for vague items is correct.

INPUT DATA:
{json.dumps(user_payload, indent=2)}"""

    try:
        response = groq_ai.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_MODEL_3},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Pipeline_Field_Validation_And_Repair",
                    "strict": True,
                    "schema": Model_3_Validation_Schema["schema"]
                }
            },
            temperature=0.0,
            max_tokens=4096
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return handle_ai_error(e)

def apply_validation_patches(model_2_output: dict, audit_report: dict) -> dict:
    """Deterministic Python application of Model 3 field corrections."""
    if not model_2_output or "tasks" not in model_2_output:
        return model_2_output

    if not audit_report or "corrections" not in audit_report:
        return model_2_output

    tasks = [dict(t) for t in model_2_output["tasks"]]
    corrections = audit_report.get("corrections", [])

    for patch in corrections:
        idx = patch.get("task_index")
        field = patch.get("field")
        new_val = patch.get("new_value")
        if idx is not None and 0 <= idx < len(tasks) and field in tasks[idx]:
            old_val = tasks[idx][field]
            tasks[idx][field] = new_val
            print(f"[Pipeline Patch] Task {idx} ('{tasks[idx].get('title')}'): "
                  f"updated '{field}' from '{old_val}' -> '{new_val}' | Reason: {patch.get('reason')}")

    return {"tasks": tasks}