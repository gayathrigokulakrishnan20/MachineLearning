import pandas as pd

INPUT_PATH = "data\\learners.csv"
OUTPUT_PATH = "data\\learners_clean.csv"

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", before - len(df))

# Remove duplicate learner IDs
before = len(df)
df = df.drop_duplicates(subset=["learner_id"])
print("Duplicate learner IDs removed:", before - len(df))

# Validate numeric ranges
df.loc[df["study_hours_per_week"] < 0, "study_hours_per_week"] = pd.NA
df.loc[df["quiz_score_pct"].between(0, 100) == False, "quiz_score_pct"] = pd.NA
df.loc[df["days_since_last_login"] < 0, "days_since_last_login"] = pd.NA
df.loc[df["lessons_completed"] < 0, "lessons_completed"] = pd.NA
df.loc[df["courses_started"] < 0, "courses_started"] = pd.NA

# Fill numeric missing values with median
numeric_columns = [
    "study_hours_per_week",
    "quiz_score_pct",
    "days_since_last_login",
    "lessons_completed",
    "courses_started",
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

df.to_csv(OUTPUT_PATH, index=False)

print("Missing values after cleaning:")
print(df.isnull().sum())
print("Clean dataset saved to:", OUTPUT_PATH)
