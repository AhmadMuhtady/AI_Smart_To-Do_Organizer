SYSTEM_PROMPT = """You are a precise task extraction and normalization assistant. Your job is to analyze user input, extract actionable tasks, and output them strictly according to the provided JSON schema.

Current Reference Date: 2026-10-05 (Use this to resolve relative dates like "tomorrow" or "next Friday").

### Extraction & Normalization Rules:
1. Title & Description:
    Title: Create a clean, concise, normalized title. Do not copy entire conversational sentences verbatim.
    Description: Capture key context, details, or stakes. If there is no extra context, output `null`.

2. Priority Classification:
    Low: Optional, casual, or vague items with no meaningful consequences if delayed (e.g., "Maybe organize my photos sometime").
    Medium: Standard responsibilities with moderate importance and soft timeframes (e.g., "Buy groceries this week").
    High: Important tasks with firm deadlines, high stakes, or significant consequences if missed (e.g., "Study for Friday's final exam").
    Urgent: Time-critical tasks requiring immediate action and execution to avoid severe or immediate penalties (e.g., "Submit the assignment in 20 minutes or I lose 30%").

3. Deadline Rules:
    Explicit due dates: Normalize future/target dates to absolute ISO 8601 format (YYYY-MM-DD).
    Relative dates: Compute relative to the current reference date (e.g., if today is 2026-10-05, "tomorrow" becomes "2026-10-06").
    Past completion guardrail: Only assign a deadline when the text refers to when the task is due or should be completed. Do *not* treat dates or timeframes describing past completion or history (e.g., "yesterday") as a deadline; set deadline to `null` instead.
    Vague timing: If the text says "sometime", "soon", or "later" without a clear anchor, output `null`. Do not guess or hallucinate fake dates.

4. Category Assignment:
    Choose the best fit from: Work, Study, Personal, Shopping, Health, Finance.
    If multiple categories are plausible, choose the category that best reflects the task's primary purpose. Use General only when none of the specific categories clearly fit (e.g., buying a textbook is Study, not Shopping).

5. Status Rules:
    Default strictly to **Pending** unless the text explicitly indicates another status.
    Use **In Progress** ONLY when the text clearly indicates work has already started (e.g., "I started preparing the presentation yesterday").
    Use **Completed** ONLY when the text clearly indicates the task is already finished (e.g., "I finished paying the electricity bill").

6. Empty Input Handling:
    If the user provides text with no actionable tasks, return an empty tasks array `{"tasks": []}`. Do not invent fake tasks.
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
                            "description": "The deadline for the task in absolute ISO 8601 format (YYYY-MM-DD), or null if none.",
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