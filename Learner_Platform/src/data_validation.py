import pandas as pd

df = pd.read_csv("data\\learners.csv")

required = [
    "learner_id", "study_hours_per_week", "quiz_score_pct",
    "days_since_last_login", "lessons_completed",
    "courses_started", "course_completed"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["learner_id"].duplicated().any():
    raise ValueError("Duplicate learner IDs found.")

if df.isnull().any().any():
    raise ValueError("Missing values found.")

if not df["course_completed"].isin([0, 1]).all():
    raise ValueError("Target must be 0 or 1.")

print("Data validation passed.")
