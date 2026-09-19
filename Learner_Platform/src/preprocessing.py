import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data\\learners_clean.csv"

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "study_hours_per_week",
    "quiz_score_pct",
    "days_since_last_login",
    "lessons_completed",
    "courses_started",
]

X = df[FEATURES]
y = df["course_completed"]

# Split BEFORE fitting preprocessing steps to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)
print("Training target distribution:")
print(y_train.value_counts(normalize=True))
print("Testing target distribution:")
print(y_test.value_counts(normalize=True))
