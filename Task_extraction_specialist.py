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
1. Extract Genuine Actionable Tasks Across All States (Completed, In-Progress, Pending):
   - A pure complaint, accidental happening, or passive narrative is NOT a task:
     * "I spilled coffee all over my desk earlier, but whatever." -> IGNORE (passive accident; no action verb/request).
     * "The weather was terrible today." -> IGNORE (observation).
     * "My brother called to tell me about his dog." -> IGNORE (narrative).
   - Do NOT ignore actions merely because they happened in the past or are already under way:
     * "I already finished updating the onboarding doc yesterday" -> MUST EXTRACT (task_text: "update the team onboarding document", state: "already finished", time: "yesterday").
     * "I've started drafting the Q4 marketing plan" -> MUST EXTRACT (task_text: "draft the Q4 marketing plan", state: "started").
   - General Extraction Rule: If there is an actionable verb expressing work (update, draft, file, clean, submit, send), extract it regardless of whether it is past, present, or future, EXCEPT when the action has been explicitly negated, retracted, cancelled, or superseded by a later instruction.

2. Future Appointments, Commitments & Scheduled Events (MANDATORY TO EXTRACT):
   - Future commitments, appointments, meetings, interviews, doctor visits, or exams that require the user's attendance or participation ARE actionable tasks, even when phrased casually as "I have [event]...":
     * "I have a job interview next week" -> MUST EXTRACT (task_text: "attend job interview", time_expression: "next week", state_expression: null, context: null).
     * "I have a dentist appointment Friday" -> MUST EXTRACT (task_text: "attend dentist appointment", time_expression: "Friday", state_expression: null, context: null).
   - Guardrail: Do NOT extract third-party events or general statements where the user is NOT participating (e.g., "There is a football match Friday" -> IGNORE unless the user states "I'm going to the football match").

3. Resolve Obvious Pronoun / Antecedent References:
   - When a task refers back to an entity from an adjacent sentence or clause using pronouns or demonstratives (e.g., "it", "that", "them", "for it"), dereference it to the unambiguous noun phrase in context:
     * "I have a job interview next week. I need to buy clothes for it."
       -> Task 1: task_text: "attend job interview", time_expression: "next week"
       -> Task 2: task_text: "buy clothes", context: "for the job interview", state_expression: "need to" (NOT "for it").

4. Preserve All Timing Expressions (Specific and Vague):
   - NEVER omit a timing phrase simply because it is non-specific or spans multiple days.
   - Phrases like "this weekend", "next week", "sometime soon", "tomorrow afternoon", "this evening" MUST be captured verbatim in `time_expression`:
     * "go to the barber this weekend" -> task_text: "go to the barber", time_expression: "this weekend".

5. Resolve Corrections, Retractions, and Replacements First (CRITICAL):
   - Before creating task objects, FIRST resolve any explicit correction, retraction, or replacement.
   - If a later clause clearly replaces an earlier instruction, output ONLY the final intended instruction.
   - If a later clause only changes timing or details of the same task, output ONE task with the corrected details.
   - A task that was explicitly cancelled, retracted, or replaced MUST NOT appear in the output.
   - Words like "actually", "wait", "instead", "rather", "no", "don't do that" are signals, but only treat them as replacements when the text clearly establishes that superseding relationship.
   - Conceptual Replacement Examples:
     * "Send Mike the project summary by Friday, actually wait, send him the updated spreadsheet tomorrow afternoon instead."
       -> Expected: ONLY "send him the updated spreadsheet", time: "tomorrow afternoon". (The project summary task was retracted/replaced).
     * "Book flight tickets next Monday, but wait, don't do that yet, book the hotel by Thursday instead."
       -> Expected: ONLY "book the hotel", time: "by Thursday".
     * "Cancel my subscription whenever, actually later this month."
       -> Expected: ONE task ("cancel my subscription"), time: "later this month".
   - Counterexample (Do NOT Replace):
     * "I need to email Sarah, and actually I should call Mike too."
       -> Expected: TWO tasks ("email Sarah" and "call Mike"), because nothing was retracted or replaced.

6. Split Multiple Independent Tasks:
   - Only AFTER all corrections, retractions, and replacements are resolved, split the remaining independent actions into task objects.
   - Execution pipeline: Resolve revisions -> then split remaining tasks -> then populate fields.

7. Timing Boundary — Task Timing vs. Consequence Timing (STRICT):
   - `time_expression` captures ONLY the timing phrase that describes WHEN the user performs the task itself.
   - Any timing describing a consequence, penalty, outcome, reason, or external event (e.g., "or power cuts in 10 minutes", "or they'll cut the power tonight", "or receive a penalty in an hour") MUST REMAIN INSIDE `context` and NEVER leak into `time_expression`.
     * "Pay the electricity bill right now or they'll cut the power tonight."
       -> `task_text`: "Pay the electricity bill"
       -> `time_expression`: "right now"
       -> `context`: "or they'll cut the power tonight"
   - For booking/reservation tasks (e.g. "book tickets for tomorrow", "schedule meeting for Friday"), the phrase "for tomorrow" / "tomorrow" describes the task execution target and belongs strictly in time_expression, NOT in task_text or context.
     * "Book cinema ticket for tomorrow" -> task_text: "Book cinema ticket", time_expression: "tomorrow", context: null.

8. Action Verb Belongs Exclusively in Task Text:
   - `task_text` MUST ALWAYS contain the actionable verb and its direct object/modifiers.
   - For appointments phrased as "I have [X]", synthesize an actionable attending verb: "attend job interview", "attend doctor appointment".

9. State Expression Boundary:
   - Contains ONLY auxiliary words indicating intent, reminders, progress, or completion ("need to", "remind me to", "already finished", "started"). Null if none.

10. Context Boundary (Include Consequence Timing & Explicit Causal Context):
   - Auxiliary reasons, locations, purposes, consequences, or penalties ("for the job interview", "or they'll cut the power tonight").
   - Timing that describes a consequence, penalty, reason, event, or outcome must remain inside context; only timing describing when the user performs the task belongs in time_expression.
   - Explicit Causal Context: Preserve explicit causal context introduced by phrases such as "because", "so that", "or else", "otherwise", and similar wording when it explains why the task matters.
     * "Call my bank today because my card keeps getting declined."
       -> task_text: "Call my bank"
       -> time_expression: "today"
       -> context: "because my card keeps getting declined"
   - Never duplicate task_text or action timing inside context. Null if none.

Do not invent missing information. If a detail is missing, output null for that field."""

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