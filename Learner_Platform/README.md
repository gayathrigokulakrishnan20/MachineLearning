# ML Book Notes #11–#13 — Deployment, Monitoring & Production Architecture

Prediction and REST API are already covered in Post #10. This package focuses on deployment, monitoring, and the end-to-end operating model.

## Existing project prerequisites
- `src/predict_api.py` — FastAPI app from Post #10
- `models/learner_completion_pipeline.joblib` — saved pipeline from Post #9
- `requirements.txt` — API runtime dependencies

Keep your existing API; do not duplicate it.

## Post #11 — Deployment
`render.yaml` and `Procfile` are examples. Adapt the start command and build settings to your repository. Ensure the trained model artifact is available in the deployed environment.

Deployment checklist:
1. Confirm the app starts locally.
2. Install runtime dependencies.
3. Ensure the model artifact is available.
4. Configure environment variables and secrets.
5. Deploy to your chosen host.
6. Test `/health` and a prediction end-to-end.
7. Enable logs, alerts, access controls, and rollback.

## Post #12 — Monitoring
Install `requirements-monitoring.txt`, then run:
`python -m src.monitor --reference data/reference.csv --current data/current.csv`

CSV files should contain one or more monitored columns such as `study_hours_per_week`, `quiz_score_pct`, `days_since_last_login`, `lessons_completed`, and `courses_started`. This is an educational screening signal, not proof of drift.

Monitor data quality, feature/prediction distributions, model performance when actual outcomes arrive, API latency, errors, throughput, and model version. Drift should trigger investigation, not automatic retraining by itself.

## Post #13 — End-to-End Architecture
See `docs/END_TO_END_ARCHITECTURE.md`.

Before production, add authentication, rate limiting, tests, structured logs, secure secrets, artifact/version management, alerting, rollback, privacy controls, and a documented retraining approval process. Verify current hosting-provider settings before deployment.
