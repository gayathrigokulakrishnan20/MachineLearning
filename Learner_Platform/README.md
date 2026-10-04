# Learning Platform ML

A production-oriented ML project built step-by-step alongside my LinkedIn ML Book Notes series.

## Goal
Build a learning-platform ML pipeline for course-completion prediction, learner-risk detection, recommendations and segmentation.

## Current stage: Step 1 — Data
The project starts with a reproducible synthetic learner dataset.

Features:
- study_hours_per_week
- quiz_score_pct
- days_since_last_login
- lessons_completed
- courses_started

Target:
- course_completed (0/1)

## Roadmap
1. Problem definition & data
2. Data exploration
3. Data validation
4. Data cleaning
5. Data preprocessing
6. Feature engineering
5. Train/test split
6. Model training
7. Evaluation
8. Model comparison
9. Model persistence
10. API
11. Docker
12. Monitoring & retraining

## Run
```bash
pip install -r requirements.txt
python src/data_exploration.py
```
