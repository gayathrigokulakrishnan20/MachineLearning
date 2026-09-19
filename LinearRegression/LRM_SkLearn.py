"""Train and evaluate scikit-learn's linear-regression implementation."""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the regression dataset and hold back 20% for evaluation.
data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the closed-form linear-regression estimator to the training examples.
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the unseen targets and display quality plus learned parameters.
y_pred = model.predict(X_test)

print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R²:", r2_score(y_test, y_pred))
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
