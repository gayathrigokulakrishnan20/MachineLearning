# Source code

## Current pipeline

1. `data_exploration.py` — understand the dataset
2. `data_validation.py` — validate schema and target
3. `data_cleaning.py` — remove duplicates, handle invalid values and missing values
4. `preprocessing.py` — split data and scale numerical features

Important production principle:

**Split first, then fit preprocessing only on training data.**

This helps prevent data leakage.
