from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_learner

app = FastAPI(
    title="Learning Platform ML Prediction API",
    version="1.0.0",
)


class LearnerRequest(BaseModel):
    study_hours_per_week: float = Field(ge=0, le=168)
    quiz_score_pct: float = Field(ge=0, le=100)
    days_since_last_login: int = Field(ge=0)
    lessons_completed: int = Field(ge=0)
    courses_started: int = Field(ge=0)


@app.get("/health")
def health() -> dict:
    return {"status": "UP"}


@app.post("/predict")
def predict(request: LearnerRequest) -> dict:
    try:
        return predict_learner(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Trained model not found. Run train_model.py first.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)
