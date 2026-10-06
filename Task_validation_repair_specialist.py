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

    # 1. Vague or past completed history -> null
    if any(k in raw for k in ["sometime", "later", "whenever", "yesterday", "last week", "past"]):
        return None

    # 2. Immediate or same-day expressions
    if "today" in raw or "right now" in raw or "in " in raw:
        return ref_date.strftime("%d-%m-%Y")

    # 3. Tomorrow
    if "tomorrow" in raw:
        return (ref_date + timedelta(days=1)).strftime("%d-%m-%Y")

    # 4. Strict Weekday Calculation
    for name, target_idx in WEEKDAYS.items():
        if name in raw:
            current_idx = ref_date.weekday()
            is_next = "next " + name in raw

            if target_idx == current_idx:
                days_ahead = 7 if is_next else 0
            else:
                nearest_offset = (target_idx - current_idx) % 7
                days_ahead = nearest_offset + 7 if is_next else nearest_offset

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
   - NO STYLISTIC REWRITING OR SIMPLIFYING (HARD RULE):
     If the candidate description already faithfully preserves the source information, do NOT rewrite, shorten, normalize, or stylistically improve it.
     * "sometime next week" -> LEAVE IT ALONE (do NOT simplify to "next week").
     * "sometime this weekend, if I get the chance" -> LEAVE IT ALONE.
     * Model 3 only repairs missing or wrong information; it is NOT a copyeditor.

   - Specific Sub-Day Timing (MANDATORY TO PRESERVE):
     Phrases specifying an exact time of day or interval ("afternoon", "in 15 minutes", "at 3 PM", "right now", "before lunch") MUST be preserved in description if missing.

   - Multi-Day / Window Timing Without Concrete Single-Day Deadlines:
     Phrases that specify an active planning timeframe that cannot be pinned to one exact calendar date (e.g., "this weekend", "next week") MUST be preserved in description if candidate deadline is null and description is empty.

   - Completely Vague or Indefinite Phrases (DO NOT PUT IN DESCRIPTION):
     Phrases with zero scheduling boundary such as "sometime soon", "later this month maybe", "whenever get a chance", or "sometime" DO NOT belong in description. Leaving description null for these is correct.

   - Calendar Deadlines (DD-MM-YYYY):
     Single calendar days ("today", "tomorrow", "Friday") captured by deadline must NOT be duplicated in description.

3. Hallucination Removal:
   - Clear any details in `description` or `title` that have zero basis in Model 1 or raw text (set `new_value: null`).

4. Status & Progress Verification:
   - Compare `status` directly against Model 1's `state_expression`.
   - Completion evidence ("already finished", "done", "finished filing") -> `status: "Completed"`.
   - Progress evidence ("started", "in progress") -> `status: "In Progress"`.
   - Otherwise -> `status: "Pending"`.

5. Priority Audit (Narrow, Rule-Based Only):
   - Only correct priority when it is an obvious rule violation. Do NOT debate subjective Medium vs. High choices.
   - Standard work, assignments, or presentations (e.g. "Presentation for Monday meeting" as Medium vs. High): LEAVE ALONE.
   - Immediate Action + Severe Imminent Consequence within hours or minutes MUST be "Urgent":
     * "pay electricity bill right now or power will be cut tonight" -> if marked Medium/High, emit repair setting `new_value: "Urgent"`.
     * "pay parking ticket in 15 minutes or receive penalty" -> must be "Urgent".

6. Category Guardrail:
   - Only correct blatant blunders (e.g., studying for an exam marked as "Shopping" instead of "Study", barber/grooming marked as "Work").
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
- Priority rule violations: immediate action + severe penalty within hours MUST be 'Urgent' (do NOT debate subjective Medium vs High)
- Missing specific sub-day timing that a date cannot represent (e.g. 'in 15 minutes', 'afternoon')
- Missing consequences, locations, or penalties
- Hallucinated details
- Incorrect status or category

Do NOT rewrite, shorten, or cosmetically edit descriptions that already faithfully capture source details (e.g. 'sometime next week' must remain 'sometime next week').
Do NOT add completely vague timing phrases ('sometime soon', 'later this month maybe', 'whenever') into descriptions. Leaving description null for those is correct.

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