from datetime import date
import os
import json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, NotFoundError

load_dotenv(override=True)
groq_ai_key = os.getenv('GROQAI_API_KEY')
groq_ai = OpenAI(api_key=groq_ai_key, base_url="https://api.groq.com/openai/v1")

from datetime import date

today = date.today()
# Use dashes to perfectly match the DD-MM-YYYY format you want the model to output
formatted_date = today.strftime("%d-%m-%Y")

SYSTEM_PROMPT = f"""You are a precise task extraction and normalization assistant. Your job is to analyze user input, extract actionable tasks, and output them strictly according to the provided JSON schema.

Current Reference Date: {formatted_date} (Use this to resolve relative dates and weekdays).

### Extraction & Normalization Rules:
1. Completeness & Extraction Scope:
    - Core Rule: No valid deadline ≠ no valid task. You must extract an actionable item even if its timing is vague, missing, or in the past. 
    - Extraction Logic:
      - Future task → extract, assign deadline.
      - No date → extract, deadline `null`.
      - Vague date (e.g., "sometime", "later") → extract, deadline `null`.
      - Past date describing completion (e.g., "yesterday") → extract, status `Completed`, deadline `null`.
      - Non-actionable chatter → ignore.
    - Multi-task split: If one sentence contains multiple distinct actionable tasks, return each one as a separate task object.

2. Title & Description:
    Title: Create a clean, concise, normalized title. Titles must be grammatically correct, properly spaced, and written as concise verb phrases (e.g., "Take car to mechanic"). Do not copy entire conversational sentences verbatim.
    Description: Capture key context, details, or stakes. Do not repeat information from the title. If there is no additional useful context, output `null`.

3. Priority Classification:
    Low: Optional, casual, or vague items with no meaningful consequences if delayed (e.g., "Maybe organize my photos sometime").
    Medium: Standard responsibilities with moderate importance and soft timeframes (e.g., "Buy groceries this week").
    High: Important tasks with firm deadlines, high stakes, or significant consequences if missed (e.g., "Study for Friday's final exam").
    Urgent: Time-critical tasks requiring immediate action and execution to avoid severe or immediate penalties (e.g., "Submit the assignment in 20 minutes or I lose 30%"). A near deadline alone does not make a task Urgent; use High instead.

4. Deadline Rules:
    Explicit due dates & weekdays: Normalize future/target dates to the format DD-MM-YYYY (e.g., 05-10-2026).
    Relative & Weekday resolution (relative to reference date {formatted_date}):
    - Immediate/Same-day: If timing is immediate (e.g., "right now", "today", "in 10 minutes"), use the current reference date as the deadline and put the precise timing/consequence in the description.
    - "tomorrow" becomes the next immediate calendar day (e.g., 06-10-2026).
    - Unqualified weekdays (e.g., "Friday") or "this Friday" resolve to the nearest upcoming occurrence of that weekday (e.g., 09-10-2026).
    - "next Friday" resolves to the Friday of the following week (e.g., 16-10-2026).
    Past completion guardrail: Only assign a deadline when the text refers to when the task is due or should be completed. Do *not* treat dates or timeframes describing past completion or history (e.g., "yesterday") as a deadline; set deadline to `null` instead.
    Vague timing: If the text says "sometime", "soon", or "later" without a clear anchor, output `null`. Do not guess or hallucinate fake dates.

5. Category Assignment:
    Choose the best fit from: Work, Study, Personal, Shopping, Health, Finance.
    If multiple categories are plausible, choose the category that best reflects the task's primary purpose. Use General only when none of the specific categories clearly fit (e.g., buying a textbook is Study, not Shopping).

6. Status Rules:
    Default strictly to **Pending** unless the text explicitly indicates another status.
    Use **In Progress** ONLY when the text clearly indicates work has already started (e.g., "I started preparing the presentation yesterday").
    Use **Completed** ONLY when the text clearly indicates the task is already finished (e.g., "I finished paying the electricity bill").

7. Empty Input Handling:
    If the user provides text with no actionable tasks, return an empty tasks array {{"tasks": []}}. Do not invent fake tasks.
"""

Smart_To_Do_Organizer_format = {
    "type": "json_schema",
    "name": "Smart_To_Do_Organizer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "A list of tasks extracted and normalized from the input text.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The clear, concise title or name of the task.",
                        },
                        "description": {
                            "type": ["string", "null"],
                            "description": "Detailed notes or context about the task.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["Low", "Medium", "High", "Urgent"],
                            "description": "Priority level based on impact and urgency rules.",
                        },
                        "deadline": {
                            "type": ["string", "null"],
                            "description": "The deadline for the task in the format DD-MM-YYYY, or null if none.",
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "Work", 
                                "Study", 
                                "Personal", 
                                "Shopping", 
                                "Health", 
                                "Finance", 
                                "General"
                            ],
                            "description": "The category the task belongs to, using General as a fallback.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["Pending", "In Progress", "Completed"],
                            "description": "The current status of the task.",
                        },
                    },
                    "required": [
                        "title", 
                        "description", 
                        "priority", 
                        "deadline", 
                        "category", 
                        "status"
                    ],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}





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



def ai_toDo_organizer(text):
    user_prompt = f"""
        Please extract and organize all actionable tasks from the following user input:
        "{text}"
        """
    try:
        response = groq_ai.responses.create(
            model = 'openai/gpt-oss-120b',
            instructions = SYSTEM_PROMPT,
            input = user_prompt,
            text={'format': Smart_To_Do_Organizer_format}
        )

        results = json.loads(response.output_text)
        return results
    except json.JSONDecodeError as e:
        return handle_ai_error(e)
    except Exception as e:
        return handle_ai_error(e)


res = ai_toDo_organizer(raw_text)
print(res)