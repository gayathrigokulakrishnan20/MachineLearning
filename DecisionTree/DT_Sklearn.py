"""Train and evaluate a bounded-depth scikit-learn decision tree."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Customer features: age, income, and visit count.
X = np.array([
    [22, 25000, 8],
    [25, 30000, 7],
    [28, 35000, 6],
    [35, 40000, 2],
    [42, 45000, 3],
    [50, 60000, 1],
])

# Binary purchase labels corresponding to the rows above.
y = np.array([1, 1, 1, 0, 0, 0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Limit depth to keep the illustrative tree simple.
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

# Fit on the training portion and assess predictions on held-out rows.
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
