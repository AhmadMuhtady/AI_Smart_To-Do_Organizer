import os
import json
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError

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

1. Title:
   - Format as a clean, concise, grammatically correct verb phrase.
   - Retain original intent without adding unstated facts.

2. Description (Anti-Duplication Rule):
   - Only preserve context or specific timing details that CANNOT fit into a calendar date (e.g., "14:30", "afternoon", "in 10 minutes or fail").
   - Do NOT duplicate relative dates that are already captured by the deadline field (e.g., do not keep "tomorrow" if deadline is set to tomorrow's date).
   - If there is no extra context beyond what is captured by title and deadline, output null.

3. Priority (Independent of Status):
   - Priority reflects the importance and consequence of the task itself, COMPLETELY INDEPENDENT of status. A completed tax return or completed critical project retains its true priority (e.g., Medium or High), NEVER Low simply because it is done.
   - Low: purely optional, discretionary, casual, or vague items easily postponed without real consequence ("organize garage sometime", "cancel subscription whenever").
   - Medium: ordinary concrete responsibilities, routine appointments, chores, or errands expected to actually be done ("buy milk today", "pick up dry cleaning tomorrow", "call dentist").
   - High: significant importance with firm deadlines or meaningful stakes/consequences ("study for final exam", "quarterly executive deliverable").
   - Urgent: immediate action needed or immediate severe penalty within hours/minutes ("pay bill right now or power cuts in 10 minutes", "submit in 5 minutes or fail course"). A nearby deadline alone is NOT Urgent.

4. Deadline & Calendar Verification:
   - Output format: DD-MM-YYYY or null.
   - "today" / immediate timing -> current date ({formatted_date}).
   - "tomorrow" -> next calendar day.
   - Weekday Resolution Rule:
     * Unqualified weekday (e.g., "Friday") or "this Friday" -> the nearest upcoming occurrence of that day.
     * "next Friday" -> the Friday of the following week (7 days after the nearest Friday).
     * MANDATORY INTERNAL CHECK: Before finalizing a date, confirm that the DD-MM-YYYY date mathematically falls on the requested weekday relative to {weekday_name} {formatted_date}.
   - Vague timing ("sometime", "later this month maybe") -> null.
   - Past completed history ("yesterday", "last week") -> null.

5. Category:
   - Select from: Work, Study, Personal, Shopping, Health, Finance, General.
   - Purpose beats shallow keywords (e.g., buying a textbook is Study, buying groceries is Shopping, dentist appointment is Health). Use General only when nothing else fits.

6. Status:
   - Base ONLY on `state_expression`.
   - Pending: default state.
   - In Progress: explicit phrase showing work has started ("started studying", "working on").
   - Completed: explicit phrase showing work is already finished ("already finished", "done").
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
                        "description": {"type": ["string", "null"], "description": "Extra context or sub-day timing not captured in deadline."},
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

def task_classification_specialist(model_1_output: dict) -> dict:
    if not model_1_output or "tasks" not in model_1_output:
        return {"error": "invalid_input", "message": "Model 1 did not return valid task data."}

    user_prompt = f"""Normalize and classify these extracted tasks. 
Output ONLY valid JSON matching this schema:
{{
  "tasks": [
    {{
      "title": "string",
      "description": "string or null",
      "priority": "Low | Medium | High | Urgent",
      "deadline": "DD-MM-YYYY or null",
      "category": "Work | Study | Personal | Shopping | Health | Finance | General",
      "status": "Pending | In Progress | Completed"
    }}
  ]
}}

INPUT DATA:
{json.dumps(model_1_output, indent=2)}
"""

    try:
        response = groq_ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_MODEL_2},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4096
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return handle_ai_error(e)