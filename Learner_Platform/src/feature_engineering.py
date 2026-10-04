import pandas as pd

INPUT_PATH = "data/learners_clean.csv"
OUTPUT_PATH = "data/learners_features.csv"
TARGET = "course_completed"

CANDIDATE_FEATURES = [
    "study_hours_per_week",
    "quiz_score_pct",
    "days_since_last_login",
    "lessons_completed",
    "courses_started",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reusable learner features from cleaned source data."""
    df = df.copy()

    df["recent_activity_score"] = 1 / (1 + df["days_since_last_login"])
    df["study_quiz_interaction"] = (
        df["study_hours_per_week"] * df["quiz_score_pct"]
    )
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    """Select candidate features using target correlation as a starting point."""
    correlation = (
        df[CANDIDATE_FEATURES + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    selected = correlation[correlation.abs() >= 0.10].index.tolist()
    return selected


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = create_features(df)

    selected = select_features(df)
    final_features = list(dict.fromkeys(
        selected + ["recent_activity_score", "study_quiz_interaction"]
    ))

    output_columns = ["learner_id"] + final_features + [TARGET]
    df[output_columns].to_csv(OUTPUT_PATH, index=False)

    print("Selected features:")
    for feature in final_features:
        print(f" - {feature}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
