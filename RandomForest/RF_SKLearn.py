"""Classify learner-completion risk with a random forest."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# A small learning-platform-style dataset.
# Features:
# study_hours_per_week, completed_lessons_pct,
# quiz_avg_pct, days_since_last_login
X = np.array([
    [12, 90, 88, 1],
    [10, 85, 82, 2],
    [8, 78, 76, 3],
    [6, 65, 68, 5],
    [5, 60, 61, 7],
    [3, 40, 45, 14],
    [2, 30, 38, 21],
    [1, 20, 30, 30],
    [11, 92, 90, 1],
    [7, 72, 70, 4],
    [4, 50, 55, 10],
    [2, 25, 35, 25],
])

# 1 = likely to complete, 0 = at risk of dropping out
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])

# Reserve a stratified quarter of the small dataset for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Combine shallow decision trees into a reproducible ensemble.
model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Train the forest, then obtain class labels and completion probabilities.
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Feature importance:", model.feature_importances_)
print("Completion probabilities:", y_probability)
