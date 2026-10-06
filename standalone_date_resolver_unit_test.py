from datetime import date
from Task_validation_repair_specialist import resolve_expected_deadline

# Tuesday 06-10-2026
ref_tuesday = date(2026, 10, 6)

test_cases = [
    ("today", "06-10-2026"),
    ("tomorrow before lunch", "07-10-2026"),
    ("Friday", "09-10-2026"),
    ("this Friday", "09-10-2026"),
    ("next Friday", "16-10-2026"),
    ("Tuesday", "06-10-2026"),
    ("next Tuesday", "13-10-2026"),
    ("sometime later this month", None),
    ("yesterday", None),
]

for expr, expected in test_cases:
    actual = resolve_expected_deadline(expr, ref_date=ref_tuesday)
    assert actual == expected, f"Failed on '{expr}': expected {expected}, got {actual}"
    print(f"✓ '{expr}' -> {actual}")

print("\nAll calendar date calculations verified.")