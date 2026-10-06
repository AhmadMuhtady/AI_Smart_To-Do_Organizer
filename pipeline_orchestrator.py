import logging
import time
from typing import Any, Dict, List, Optional
from text_validation import text_validation
from Task_extraction_specialist import task_extraction_specialist
from Task_classification_specialist import task_classification_specialist
from Task_validation_repair_specialist import (
    validate_and_repair_tasks,
    apply_validation_patches,
)

logger = logging.getLogger("TaskPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PipelineExecutionError(Exception):
    """Raised when an unrecoverable failure occurs inside the pipeline."""
    pass


def _execute_with_retry(
    callable_fn,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.5,
    stage_name: str = "Stage",
    **kwargs,
) -> Dict[str, Any]:
    """Executes a specialist stage with exponential backoff for transient API faults."""
    delay = initial_delay
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            result = callable_fn(*args, **kwargs)
            if not isinstance(result, dict):
                raise PipelineExecutionError(
                    f"{stage_name} returned non-dict payload: {type(result)}"
                )


            if "error" in result:
                err_type = result.get("error")
                err_detail = result.get("detail", result.get("message", "Unknown error"))
                logger.warning(
                    f"[{stage_name}] Attempt {attempt}/{max_retries} encountered '{err_type}': {err_detail}"
                )
                last_error = f"{err_type}: {err_detail}"
            else:
                return result

        except Exception as exc:
            logger.warning(
                f"[{stage_name}] Attempt {attempt}/{max_retries} raised exception: {exc}"
            )
            last_error = str(exc)

        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2.0

    raise PipelineExecutionError(
        f"{stage_name} failed after {max_retries} attempts. Last error: {last_error}"
    )


def process_user_text_to_tasks(
    raw_text: str,
    max_classification_retries: int = 2,
    rate_limit_pause: float = 1.0,
) -> List[Dict[str, Any]]:
    """End-to-end task extraction, normalization, and repair pipeline.

    Args:
        raw_text: Messy input string from the user.
        max_classification_retries: Number of full-batch retries for Model 2 cardinality mismatches.
        rate_limit_pause: Inter-model pause in seconds to pace API quotas.

    Returns:
        A list of clean, normalized, and audited task dictionaries.
    """
    logger.info("Initiating task processing pipeline.")


    clean_text = text_validation(raw_text)
    if not clean_text or not clean_text.strip():
        logger.info("Empty input detected; returning empty task list.")
        return []


    logger.info("Executing Model 1 (Extraction Specialist)...")
    m1_payload = _execute_with_retry(
        task_extraction_specialist,
        clean_text,
        stage_name="Model 1 (Extraction)",
    )
    extracted_tasks = m1_payload.get("tasks", [])
    total_extracted = len(extracted_tasks)
    logger.info(f"Model 1 extracted {total_extracted} raw task objects.")

    if total_extracted == 0:
        return []

    time.sleep(rate_limit_pause)


    logger.info("Executing Model 2 (Classification & Normalization Specialist)...")
    m2_payload = None
    classified_tasks: List[Dict[str, Any]] = []

    for attempt in range(1, max_classification_retries + 1):
        m2_payload = _execute_with_retry(
            task_classification_specialist,
            m1_payload,
            stage_name=f"Model 2 (Classification, pass {attempt})",
        )
        classified_tasks = m2_payload.get("tasks", [])

        # Programmatic Cardinality Invariant Check
        if len(classified_tasks) == total_extracted:
            logger.info(
                f"Model 2 cardinality verified: {len(classified_tasks)}/{total_extracted} tasks."
            )
            break
        else:
            logger.warning(
                f"Model 2 task count mismatch (Expected: {total_extracted}, Got: {len(classified_tasks)}). "
                f"Attempt {attempt}/{max_classification_retries}."
            )
            if attempt < max_classification_retries:
                time.sleep(rate_limit_pause * 1.5)

    if len(classified_tasks) != total_extracted:
        raise PipelineExecutionError(
            f"Cardinality alignment failed: Model 1 produced {total_extracted} tasks, "
            f"but Model 2 returned {len(classified_tasks)} tasks."
        )

    time.sleep(rate_limit_pause)


    logger.info("Executing Model 3 (Audit & Validation Specialist)...")
    audit_report = _execute_with_retry(
        validate_and_repair_tasks,
        clean_text,
        m1_payload,
        m2_payload,
        stage_name="Model 3 (Validation & Repair)",
    )

    passed = audit_report.get("passed", False)
    issues_found = audit_report.get("issues_found", 0)
    corrections = audit_report.get("corrections", [])

    logger.info(
        f"Model 3 audit completed: passed={passed}, issues_found={issues_found}, corrections={len(corrections)}"
    )

    # --- Stage 4: Deterministic Runtime Patch Application ---
    repaired_payload = apply_validation_patches(m2_payload, audit_report)
    final_tasks = repaired_payload.get("tasks", [])

    if len(final_tasks) != total_extracted:
        raise PipelineExecutionError(
            f"Invariant violation post-repair: Expected {total_extracted} tasks, but got {len(final_tasks)}."
        )

    logger.info(f"Pipeline finished successfully. Yielded {len(final_tasks)} final tasks.")
    return final_tasks