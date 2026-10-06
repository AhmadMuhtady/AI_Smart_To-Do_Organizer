import json
import time
from text_validation import text_validation
from Task_extraction_specialist import task_extraction_specialist
from Task_classification_specialist import task_classification_specialist
from Task_validation_repair_specialist import (
    validate_and_repair_tasks,
    apply_validation_patches,
)

MODEL_CALL_DELAY_SECONDS = 2

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
Call dentist today.
Study for exam sometime next month.
Submit assignment next Friday.
Finish presentation for meeting.
Finish presentation for CEO meeting tomorrow.
Buy textbook tomorrow.
Buy groceries tomorrow.
Already finished an important client report.
Started cleaning my room.
Pay parking ticket in 15 minutes or receive a penalty.
"""

def run_pipeline_test():
    print("=" * 80)
    print("STAGE 0: INPUT TEXT VALIDATION")
    print("=" * 80)
    clean_text = text_validation(raw_input_text)
    print("Input validated successfully.\n")

    # -------------------------------------------------------------
    # STAGE 1: MODEL 1 (Task Extraction)
    # -------------------------------------------------------------
    print("=" * 80)
    print("STAGE 1: MODEL 1 (RAW EXTRACTION SPECIALIST)")
    print("=" * 80)
    model_1_output = task_extraction_specialist(clean_text)
    
    # Print raw output first so errors are visible immediately
    print(json.dumps(model_1_output, indent=2))
    
    if "error" in model_1_output:
        raise RuntimeError(f"Model 1 API failed: {model_1_output.get('detail', model_1_output)}")

    m1_tasks = model_1_output.get("tasks", [])
    print(f"\nModel 1 extracted: {len(m1_tasks)} tasks (Expected: 20)")
    assert len(m1_tasks) == 20, f"Model 1 failed extraction! Got {len(m1_tasks)}, expected 20."

    print(f"\nWaiting {MODEL_CALL_DELAY_SECONDS}s before calling Model 2...")
    time.sleep(MODEL_CALL_DELAY_SECONDS)

    # -------------------------------------------------------------
    # STAGE 2: MODEL 2 (Normalization & Classification)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: MODEL 2 (NORMALIZATION & CLASSIFICATION SPECIALIST)")
    print("=" * 80)
    model_2_output = task_classification_specialist(model_1_output)
    print(json.dumps(model_2_output, indent=2))
    
    if "error" in model_2_output:
        raise RuntimeError(f"Model 2 API failed: {model_2_output.get('detail', model_2_output)}")

    m2_tasks = model_2_output.get("tasks", [])
    print(f"\nModel 2 classified: {len(m2_tasks)} tasks (Expected: 20)")
    assert len(m2_tasks) == len(m1_tasks), (
        f"Cardinality mismatch! Model 1 had {len(m1_tasks)}, but Model 2 produced {len(m2_tasks)}."
    )

    print(f"\nWaiting {MODEL_CALL_DELAY_SECONDS}s before calling Model 3...")
    time.sleep(MODEL_CALL_DELAY_SECONDS)

    # -------------------------------------------------------------
    # STAGE 3: MODEL 3 (Audit & Validation)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3: MODEL 3 (FINAL VALIDATION & FIELD AUDIT)")
    print("=" * 80)
    audit_report = validate_and_repair_tasks(
        clean_text, model_1_output, model_2_output
    )
    print(json.dumps(audit_report, indent=2))
    
    if "error" in audit_report:
        raise RuntimeError(f"Model 3 API failed: {audit_report.get('detail', audit_report)}")

    # -------------------------------------------------------------
    # STAGE 4: DETERMINISTIC RUNTIME PATCH APPLICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 4: FINAL DATASET INTEGRATION")
    print("=" * 80)
    final_output = apply_validation_patches(model_2_output, audit_report)
    final_tasks = final_output.get("tasks", [])
    print(f"Final validated tasks count: {len(final_tasks)}")
    print(json.dumps(final_output, indent=2))

    # -------------------------------------------------------------
    # FINAL SANITY ASSERTIONS
    # -------------------------------------------------------------

    print("\n" + "=" * 80)
    print("STAGE 5: VERIFICATION SUMMARY")
    print("=" * 80)
    issues_found = audit_report.get("issues_found", -1)
    corrections = audit_report.get("corrections", [])

    print(f"Issues Repaired by Model 3: {issues_found}")
    print(f"Final Task Count:           {len(final_tasks)} (Expected: 20)")

    # Assert structural integrity
    assert len(final_tasks) == 20, f"Task count mismatch! Got {len(final_tasks)}"

    # Assert that Model 3 successfully caught and patched the sub-day timing
    task_3 = final_tasks[3]   # Quarterly report
    task_19 = final_tasks[19] # Parking ticket
    
    assert task_3["description"] == "afternoon", f"Task 3 missing afternoon: {task_3}"
    assert "15 minutes" in (task_19["description"] or ""), f"Task 19 missing 15 minutes: {task_19}"

    print("\n🎉 ALL PIPELINE INTEGRATION CHECKS PASSED: Extraction, Normalization, and Repair are 100% verified.")
    return final_output

if __name__ == "__main__":
    run_pipeline_test()