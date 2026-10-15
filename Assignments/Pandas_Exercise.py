"""Pandas exercise and quick reference.

This script demonstrates common pandas operations using Insurance_data.csv.
It expects the CSV file beside this script. The examples print results and do
not modify the source CSV.
"""

from pathlib import Path
import pandas as pd


DATA_FILE = Path(__file__).with_name("Insurance_data.csv")


def load_data(file_path: Path = DATA_FILE) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def main() -> None:
    # 1. Read data and inspect its structure.
    data = load_data()
    print("\n--- First rows ---")
    print(data.head())
    print("\n--- Last rows ---")
    print(data.tail())
    print("\n--- Random sample ---")
    print(data.sample(min(5, len(data)), random_state=1))
    print("\n--- Shape, columns, and index ---")
    print("shape:", data.shape)
    print("columns:", data.columns.tolist())
    print("index:", data.index)
    print("\n--- Data types and summary ---")
    data.info()
    print(data.describe(include="all"))

    # 2. Select columns, rows, and values.
    print("\n--- Select columns ---")
    print(data.iloc[:5, :3])       # by integer position
    print(data.loc[:4, data.columns[:3]])  # by index and column labels
    if "Location" in data.columns:
        print(data.get("Location", "Column not found").head())

    # 3. Missing values and duplicate rows.
    print("\n--- Missing-value counts ---")
    print(data.isna().sum())
    print("\n--- Missing-value percentages ---")
    print(data.isna().mean().mul(100).round(2))
    print("duplicate rows:", data.duplicated().sum())
    print("\n--- Example missing-value handling (on a copy) ---")
    cleaned = data.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    if len(numeric_columns):
        cleaned[numeric_columns] = cleaned[numeric_columns].fillna(
            cleaned[numeric_columns].median()
        )
    print(cleaned.isna().sum())
    # Other useful options: dropna(), fillna(value), interpolate().

    # 4. Filter, sort, and rank rows.
    print("\n--- Filter and sort ---")
    numeric_columns = data.select_dtypes(include="number").columns
    if len(numeric_columns):
        number_column = numeric_columns[0]
        print(data[data[number_column] > data[number_column].median()].head())
        print(data.sort_values(number_column, ascending=False).head())
        print(data[number_column].rank().head())
    if "Location" in data.columns:
        first_location = data["Location"].iloc[0]
        print(data[data["Location"] == first_location].head())

    # 5. Counts, aggregation, and grouping.
    print("\n--- Value counts and unique values ---")
    for column in data.select_dtypes(include="object").columns[:2]:
        print(f"{column} unique values:", data[column].nunique())
        print(data[column].value_counts(dropna=False).head())
    if "Location" in data.columns:
        print("\n--- Group by Location ---")
        print(data.groupby("Location").size().sort_values(ascending=False))
        print(data.groupby("Location").agg(["count"]).head())
        numeric_columns = data.select_dtypes(include="number").columns
        if len(numeric_columns):
            print(data.groupby("Location")[numeric_columns].agg(["mean", "min", "max"]))

    # 6. Create, rename, transform, and remove columns.
    print("\n--- Column operations ---")
    changed = data.copy()
    changed.columns = changed.columns.str.strip()
    if "insurance_cost" in changed.columns:
        changed["insurance_cost_k"] = changed["insurance_cost"] / 1000
        print(changed[["insurance_cost", "insurance_cost_k"]].head())
    print(changed.rename(columns={"Location": "City"}).head())
    if "Gender" in changed.columns:
        print("Drop a column on a copy:")
        print(changed.drop(columns=["Gender"]).head())

    # 7. String and categorical data.
    print("\n--- String and category operations ---")
    for column in data.select_dtypes(include="object").columns[:1]:
        text_values = data[column].astype("string").str.strip()
        print(text_values.str.lower().head())
        print("contains 'a':", text_values.str.contains("a", case=False, na=False).sum())
        print("as category:", data[column].astype("category").dtype)

    # 8. Reshape data with pivot tables and crosstabs.
    if "Location" in data.columns:
        print("\n--- Crosstab: Location by another category ---")
        category_columns = [
            column for column in data.select_dtypes(include="object").columns
            if column != "Location"
        ]
        if category_columns:
            print(pd.crosstab(data["Location"], data[category_columns[0]]))
        if "insurance_cost" in data.columns:
            print(pd.pivot_table(
                data, index="Location", values="insurance_cost",
                aggfunc=["mean", "median", "count"],
            ))

    # 9. Combine DataFrames (small standalone examples).
    print("\n--- Concatenate and merge examples ---")
    first_rows = data.head(2)
    next_rows = data.iloc[2:4]
    print(pd.concat([first_rows, next_rows], axis=0))
    if "Location" in data.columns:
        location_counts = data["Location"].value_counts().rename_axis("Location").reset_index(name="count")
        print(data[["Location"]].head().merge(location_counts, on="Location", how="left"))

    # 10. Convert types and export results (export examples are commented out).
    print("\n--- Type conversion ---")
    print(data.convert_dtypes().dtypes)
    # data.to_csv("insurance_data_cleaned.csv", index=False)
    # data.to_excel("insurance_data.xlsx", index=False)
    # data.to_json("insurance_data.json", orient="records", indent=2)

    # Original notebook exercises, expressed as reusable pandas operations.
    print("\n--- Original exercise results ---")
    print("Missing values by column:")
    print(data.isnull().sum())
    without_gender = data.drop(columns=["Gender"], errors="ignore")
    if "Location" in without_gender.columns:
        print("Most common locations:")
        print(without_gender["Location"].value_counts().head(5))
        print("Rows grouped and counted by location:")
        print(without_gender.groupby("Location").count().sort_index())


if __name__ == "__main__":
    main()
