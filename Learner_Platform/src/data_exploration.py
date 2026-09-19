import pandas as pd

df = pd.read_csv("data\\learners.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nTarget distribution:")
print(df["course_completed"].value_counts())

print("\nSummary:")
print(df.describe())
