"""Train and evaluate a linear regression model for insurance charges."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


class InsuranceCostAnalysis:
    """Load, prepare, analyze, and model insurance cost data."""

    FEATURES = [
        "children",
        "Claim_Amount",
        "past_consultations",
        "Hospital_expenditure",
        "Anual_Salary",
    ]
    OUTLIER_COLUMNS = [
        "bmi",
        "past_consultations",
        "Hospital_expenditure",
        "Anual_Salary",
    ]
    HIGH_VIF_COLUMNS = [
        "num_of_steps",
        "age",
        "bmi",
        "NUmber_of_past_hospitalizations",
    ]

    def __init__(self, data_path="new_insurance_data.csv", target="charges"):
        self.data_path = Path(data_path)
        self.target = target
        self.data = None
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.predictions = None

    def load_data(self):
        """Read the CSV and return the loaded dataframe."""
        self.data = pd.read_csv(self.data_path)
        return self.data

    def prepare_data(self):
        """Fill missing values and remove numeric outliers by the IQR rule."""
        if self.data is None:
            self.load_data()

        for column in self.data.columns:
            if self.data[column].isna().any():
                if pd.api.types.is_numeric_dtype(self.data[column]):
                    self.data[column] = self.data[column].fillna(self.data[column].mean())
                else:
                    mode = self.data[column].mode()
                    if not mode.empty:
                        self.data[column] = self.data[column].fillna(mode.iloc[0])

        for column in self.OUTLIER_COLUMNS:
            if column in self.data.columns and pd.api.types.is_numeric_dtype(self.data[column]):
                q1 = self.data[column].quantile(0.25)
                q3 = self.data[column].quantile(0.75)
                iqr = q3 - q1
                self.data = self.data[self.data[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

        # These predictors had high multicollinearity in the original analysis.
        self.data = self.data.drop(
            columns=[c for c in self.HIGH_VIF_COLUMNS if c in self.data.columns],
            errors="ignore",
        )
        return self.data

    def calculate_vif(self):
        """Return variance inflation factors for numeric predictors."""
        if self.data is None:
            self.prepare_data()
        predictors = self.data.select_dtypes(include="number").drop(
            columns=[self.target], errors="ignore"
        )
        if predictors.empty:
            raise ValueError("No numeric predictor columns are available for VIF.")
        return pd.DataFrame({
            "feature": predictors.columns,
            "VIF": [
                variance_inflation_factor(predictors.to_numpy(), i)
                for i in range(len(predictors.columns))
            ],
        })

    def train(self, test_size=0.2, random_state=12):
        """Split the prepared data and fit a linear regression model."""
        if self.data is None:
            self.prepare_data()
        missing = [column for column in self.FEATURES + [self.target] if column not in self.data.columns]
        if missing:
            raise ValueError(f"Required columns missing from dataset: {', '.join(missing)}")

        X = self.data[self.FEATURES]
        y = self.data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        return self.model

    def evaluate(self):
        """Return test-set predictions and common regression metrics."""
        if self.model is None:
            self.train()
        errors = pd.DataFrame({
            "Actual_data": self.y_test,
            "Prediction_data": self.predictions,
        })
        errors["Error"] = errors["Actual_data"] - errors["Prediction_data"]
        metrics = {
            "r2": r2_score(self.y_test, self.predictions),
            "rmse": mean_squared_error(self.y_test, self.predictions) ** 0.5,
            "mape": mean_absolute_percentage_error(self.y_test, self.predictions),
        }
        return metrics, errors

    def run(self):
        """Run preparation, model fitting, and evaluation; return both results."""
        self.prepare_data()
        self.train()
        return self.evaluate()


def main():
    analysis = InsuranceCostAnalysis()
    metrics, errors = analysis.run()
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nPrediction errors:")
    print(errors.head())


if __name__ == "__main__":
    main()
