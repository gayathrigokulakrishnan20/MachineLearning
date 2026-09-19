"""Fit a linear-regression model with batch gradient descent."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the bundled regression dataset and separate predictors from targets.
data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start with a zero-valued linear model and configure gradient descent.
weights = np.zeros(X_train.shape[1])
bias = 0.0
learning_rate = 0.01
epochs = 1000
n = len(X_train)

for epoch in range(epochs):
    # Compute predictions and the mean-squared-error gradients.
    predictions = X_train @ weights + bias
    error = predictions - y_train

    dw = (2 / n) * (X_train.T @ error)
    db = (2 / n) * np.sum(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    # Report training loss periodically to show convergence.
    if epoch % 100 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch}: MSE = {loss:.4f}")

# Evaluate the trained model on data it did not see during training.
y_pred = X_test @ weights + bias

print("\nTest MSE:", mean_squared_error(y_test, y_pred))
print("Test R²:", r2_score(y_test, y_pred))
