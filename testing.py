import json
import time
from text_validation import text_validation
from Task_extraction_specialist import task_extraction_specialist
from Task_classification_specialist import task_classification_specialist

# Configurable pause in seconds between model API calls
MODEL_CALL_DELAY_SECONDS = 3

# 1. RAW TEST INPUT
raw_input_text = """
Buy milk from the store today.
I need to finish my history essay by Friday and also pick up my dry cleaning tomorrow.
Submit the quarterly report tomorrow afternoon.
Sometime soon I should probably organize my garage.
Take the car to the mechanic for an oil change.
Pay the electricity bill right now or they will shut off power in 10 minutes!
I already finished filing my taxes yesterday.
I've started working on the slide deck for the client meeting.
Man, what a crazy weekend. The weather was amazing.
Oh wait, remind me to cancel that subscription whenever I get a chance, but actually do it later this month maybe, well whatever.
"""


def run_test_pipeline():
    # Step 0: Raw Input
    print("=" * 60)
    print("STEP 0: RAW INPUT")
    print("=" * 60)
    print(raw_input_text.strip())

    # Step 1: Input Validation
    clean_text = text_validation(raw_input_text)

    # Step 2: Model 1 (Task Extraction)
    print("\n" + "=" * 60)
    print("STEP 1: MODEL 1 RETURN (RAW EXTRACTION)")
    print("=" * 60)
    model_1_output = task_extraction_specialist(clean_text)
    print(json.dumps(model_1_output, indent=2))

    # Wait before calling Model 2
    print("\n" + "-" * 60)
    print(f"Waiting {MODEL_CALL_DELAY_SECONDS} seconds before calling Model 2...")
    print("-" * 60)
    time.sleep(MODEL_CALL_DELAY_SECONDS)

    # Step 3: Model 2 (Normalization & Classification)
    print("\n" + "=" * 60)
    print("STEP 2: MODEL 2 RETURN (NORMALIZATION & CLASSIFICATION)")
    print("=" * 60)
    model_2_output = task_classification_specialist(model_1_output)
    print(json.dumps(model_2_output, indent=2))

    return model_1_output, model_2_output


if __name__ == "__main__":
    run_test_pipeline()