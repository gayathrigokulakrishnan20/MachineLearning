"""Train and evaluate scikit-learn logistic regression on cancer data."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load labels and measurements, preserving class balance in each split.
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization helps the optimizer treat all features on comparable scales.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Allow ample iterations for the solver to converge on the scaled data.
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Show hard-label accuracy alongside probability estimates.
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("First 5 probabilities:\n", y_probability[:5])
