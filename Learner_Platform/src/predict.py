import joblib
import pandas as pd

MODEL_PATH = "models/learner_completion_pipeline.joblib"


def prepare_learner(data: dict) -> pd.DataFrame:
    """Apply the same feature creation logic used before training."""
    df = pd.DataFrame([data])
    df["recent_activity_score"] = 1 / (1 + df["days_since_last_login"])
    df["study_quiz_interaction"] = (
        df["study_hours_per_week"] * df["quiz_score_pct"]
    )
    return df


def predict_learner(data: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    learner = prepare_learner(data)

    prediction = int(model.predict(learner)[0])
    probability = float(model.predict_proba(learner)[0][1])

    return {
        "prediction": prediction,
        "completion_probability": round(probability, 4),
    }


if __name__ == "__main__":
    sample = {
        "study_hours_per_week": 8,
        "quiz_score_pct": 78,
        "days_since_last_login": 2,
        "lessons_completed": 15,
        "courses_started": 2,
    }

    print(predict_learner(sample))
