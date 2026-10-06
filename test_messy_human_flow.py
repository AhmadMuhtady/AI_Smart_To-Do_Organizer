import json
from pipeline_orchestrator import process_user_text_to_tasks

messy_input = """
Hey, I had such a chaotic morning today, the traffic was completely unbearable. Anyway, remind me to email Sarah the updated client proposal by Friday afternoon, but actually make sure to send her the draft slide deck tomorrow before lunch instead. Also, I really need to call the plumber right now or our basement is going to flood!

My brother called earlier to tell me about his new dog, which was cute. 

I've already started preparing the quarterly budget spreadsheet yesterday, but I still need to submit the final invoice to accounting next Tuesday. Sometime later this month I should probably renew my passport, well whenever I get around to it. Oh, and pick up my prescription from the pharmacy today on the way home.
"""

print("=" * 80)
print("TESTING FULL PIPELINE ON MESSY HUMAN INPUT")
print("=" * 80)

final_tasks = process_user_text_to_tasks(messy_input)

print("\n" + "=" * 80)
print(f"FINAL AUDITED TASKS ({len(final_tasks)} items)")
print("=" * 80)
print(json.dumps(final_tasks, indent=2))

# -------------------------------------------------------------
# Assertions
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("VERIFYING SPECIFICATION CONTRACTS")
print("=" * 80)

# Contract 1: Exactly 6 tasks (proposal task superseded by slide deck)
assert len(final_tasks) == 6, f"Expected exactly 6 tasks, but got {len(final_tasks)}"
print("✓ Task count is exactly 6 (Sarah proposal was superseded).")

# Contract 2: Slide deck task contains 'before lunch' in description and tomorrow's date
slide_task = next(t for t in final_tasks if "slide deck" in t["title"].lower())
assert slide_task["deadline"] == "07-10-2026", f"Wrong deadline for slide deck: {slide_task['deadline']}"
assert "before lunch" in (slide_task["description"] or "").lower(), f"Sub-day timing missing: {slide_task['description']}"
print("✓ Slide deck task correctly has deadline 07-10-2026 and description 'before lunch'.")

# Contract 3: Plumber task is Urgent with sub-day timing preserved
plumber_task = next(t for t in final_tasks if "plumber" in t["title"].lower())
assert plumber_task["priority"] == "Urgent", f"Expected Urgent priority, got {plumber_task['priority']}"
assert "flood" in (plumber_task["description"] or "").lower(), f"Consequence missing from plumber task: {plumber_task['description']}"
print("✓ Plumber task correctly classified as Urgent with flooding consequence preserved.")

# Contract 4: Budget spreadsheet status is 'In Progress' with null deadline
budget_task = next(t for t in final_tasks if "budget" in t["title"].lower())
assert budget_task["status"] == "In Progress", f"Expected 'In Progress', got {budget_task['status']}"
assert budget_task["deadline"] is None, f"Past task should have null deadline, got {budget_task['deadline']}"
print("✓ Budget task correctly marked 'In Progress' with null deadline.")

# Contract 5: Invoice task deadline is 13-10-2026 (next Tuesday)
invoice_task = next(t for t in final_tasks if "invoice" in t["title"].lower())
assert invoice_task["deadline"] == "13-10-2026", f"Expected '13-10-2026' for next Tuesday, got {invoice_task['deadline']}"
print("✓ Invoice task correctly assigned 13-10-2026 for 'next Tuesday'.")

# Contract 6: Passport renewal has null deadline and valid discretionary/routine priority
passport_task = next(t for t in final_tasks if "passport" in t["title"].lower())
assert passport_task["priority"] in ["Low", "Medium"], f"Unexpected priority: {passport_task['priority']}"
assert passport_task["deadline"] is None, f"Vague timing should have null deadline, got {passport_task['deadline']}"
print(f"✓ Passport task correctly assigned priority '{passport_task['priority']}' with null deadline.")

# Contract 7: Prescription is Health category with pharmacy and route context preserved
rx_task = next(t for t in final_tasks if "prescription" in t["title"].lower())
assert rx_task["category"] == "Health", f"Expected Health category, got {rx_task['category']}"
full_task_text = f"{rx_task['title']} {rx_task.get('description') or ''}".lower()
assert "pharmacy" in full_task_text, f"Pharmacy location missing from task: {rx_task}"
assert "way home" in full_task_text, f"Route context missing from task: {rx_task}"
print("✓ Prescription task correctly assigned Health category with location and route context preserved.")

print("\n🎉 ALL 7 INTEGRATION VERIFICATIONS PASSED.")