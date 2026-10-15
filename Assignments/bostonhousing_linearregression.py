"""Fit and evaluate a one-feature Boston house price regression model."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


class BostonHousePriceRegression:
    """Predict median home value (``medv``) from ``lstat``."""

    def __init__(self, data_path="Boston1.csv", test_size=0.2, random_state=123):
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.model = None

    def load_data(self):
        """Load the CSV and validate the columns used by the model."""
        self.data = pd.read_csv(self.data_path)
        required = {"lstat", "medv"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
        return self.data

    def split_data(self):
        """Create reproducible training and test partitions."""
        if self.data is None:
            self.load_data()
        X = self.data[["lstat"]]
        y = self.data["medv"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self):
        """Fit the linear regression model and return it."""
        if self.X_train is None:
            self.split_data()
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self, X=None):
        """Predict values for supplied rows, or for the held-out test set."""
        if self.model is None:
            self.fit()
        if X is None:
            X = self.X_test
        return pd.Series(self.model.predict(X), index=X.index, name="Predicted")

    def evaluate(self, X=None, y=None):
        """Return MAE, MSE, and RMSE for supplied data or the test partition."""
        if X is None or y is None:
            if self.X_test is None:
                self.split_data()
            X, y = self.X_test, self.y_test
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        return {
            "mae": mean_absolute_error(y, predictions),
            "mse": mse,
            "rmse": mse ** 0.5,
        }

    def plot_data(self):
        """Display the relationship between lstat and medv."""
        if self.data is None:
            self.load_data()
        import matplotlib.pyplot as plt

        self.data.plot(x="lstat", y="medv", style="o")
        plt.xlabel("lstat")
        plt.ylabel("medv")
        plt.show()

    def run(self):
        """Fit the model and return train and test metrics."""
        self.split_data()
        self.fit()
        return {
            "intercept": float(self.model.intercept_),
            "coefficient": float(self.model.coef_[0]),
            "train": self.evaluate(self.X_train, self.y_train),
            "test": self.evaluate(self.X_test, self.y_test),
        }


def main():
    analysis = BostonHousePriceRegression()
    results = analysis.run()
    print(f"Intercept: {results['intercept']:.4f}")
    print(f"Coefficient: {results['coefficient']:.4f}")
    for split in ("train", "test"):
        print(f"\n{split.title()} metrics:")
        for metric, value in results[split].items():
            print(f"{metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
