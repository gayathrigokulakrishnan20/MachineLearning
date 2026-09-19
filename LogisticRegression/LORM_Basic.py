"""Fit a binary logistic-regression model with gradient descent."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Use the bundled binary-classification dataset.
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling makes gradient descent much more stable for this dataset.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize a linear decision boundary before optimization.
weights = np.zeros(X_train.shape[1])
bias = 0.0
learning_rate = 0.01
epochs = 2000
n = len(X_train)

def sigmoid(z):
    # Clip inputs first so exponentiation stays numerically stable.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    # Convert linear scores to probabilities and calculate log-loss gradients.
    z = X_train @ weights + bias
    probability = sigmoid(z)

    error = probability - y_train
    dw = (X_train.T @ error) / n
    db = np.mean(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    # Monitor binary cross-entropy during training.
    if epoch % 200 == 0:
        loss = -np.mean(
            y_train * np.log(probability + 1e-9)
            + (1 - y_train) * np.log(1 - probability + 1e-9)
        )
        print(f"Epoch {epoch}: Log Loss = {loss:.4f}")

# Classify held-out examples using the conventional 0.5 probability threshold.
test_probability = sigmoid(X_test @ weights + bias)
y_pred = (test_probability >= 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
