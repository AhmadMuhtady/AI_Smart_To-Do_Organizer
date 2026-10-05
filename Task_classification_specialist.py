import os
import json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError
from Task_classification_specialist import Task_


load_dotenv(override=True)
groq_ai_key = os.getenv('GROQAI_API_KEY')
groq_ai = OpenAI(api_key=groq_ai_key, base_url="https://api.groq.com/openai/v1")

from datetime import date

today = date.today()
formatted_date = today.strftime("%d-%m-%Y")

SYSTEM_PROMPT_MODEL_2 = f"""You are Model 2: Task Normalization & Classification Specialist.
Your input is a raw JSON array of extracted tasks. Your job is to analyze the raw fields (`task_text`, `context`, `time_expression`, `state_expression`) and generate a finalized, normalized task list.

Current Reference Date: {formatted_date} (Use this to resolve relative dates).

### Normalization & Classification Rules:

1. Title:
   - Make it short and clean. Keep the original meaning.
   - Fix awkward wording/spacing. Formulate as a clean verb phrase.
   - Do NOT invent extra details.

2. Description:
   - Preserve useful extra context from the raw input.
   - Preserve time details that cannot fit in a date field (e.g., "afternoon", "in 10 minutes").
   - Do NOT simply repeat the title. Use `null` if there is no useful extra information.

3. Priority:
   - Low: optional / vague / non-important.
   - Medium: normal responsibility.
   - High: important + firm deadline or high consequence.
   - Urgent: immediate action or severe/immediate consequence. A nearby deadline alone should NOT automatically mean Urgent.

4. Deadline (Format as DD-MM-YYYY):
   - today / immediate timing (right now, in 10 minutes) → current date ({formatted_date}).
   - tomorrow → next calendar day.
   - weekday (e.g., "Friday") / "this Friday" → nearest upcoming occurrence of that weekday.
   - "next Friday" → Friday of the following week.
   - vague timing (sometime, later) → `null`.
   - past timing describing completed history (e.g., "yesterday") → `null`.

5. Category:
   - Choose ONE: Work, Study, Personal, Shopping, Health, Finance, General.
   - Base your choice on the task’s underlying purpose, not just shallow keywords. Use General if nothing else fits cleanly.

6. Status (Base this ONLY on `state_expression`):
   - Pending: default state.
   - In Progress: if state expression shows work has already started.
   - Completed: if state expression clearly shows it is already finished.
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
                        "title": {
                            "type": "string",
                            "description": "Short, clean, grammatically correct verb phrase.",
                        },
                        "description": {
                            "type": ["string", "null"],
                            "description": "Useful extra context or precise time details. Null if none.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["Low", "Medium", "High", "Urgent"],
                            "description": "Priority level based on urgency and consequence.",
                        },
                        "deadline": {
                            "type": ["string", "null"],
                            "description": "Normalized date in DD-MM-YYYY format, or null.",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["Work", "Study", "Personal", "Shopping", "Health", "Finance", "General"],
                            "description": "Task category based on primary purpose.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["Pending", "In Progress", "Completed"],
                            "description": "Current status based on state expression.",
                        },
                    },
                    "required": ["title", "description", "priority", "deadline", "category", "status"],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}



USER_PROMPT_MODEL_2 = f"""Please normalize and classify the following extracted tasks according to your system instructions:

INPUT DATA:
{model_1_json_string}
"""