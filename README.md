# ML From Scratch → scikit-learn

A practical machine learning learning journey built alongside a LinkedIn ML Book Notes series.

Each algorithm is explored in two ways:
1. Understand the mathematics and learning logic.
2. Implement the core idea with Python/NumPy.
3. Compare with the production-friendly scikit-learn implementation.

## Algorithms

- 01 Linear Regression
- 02 Logistic Regression
- 03 Decision Tree
- 04 Random Forest

## Learning-platform use cases

Examples are framed around a fictional learning platform:
- Predict course completion time
- Predict course completion / dropout risk
- Classify learner outcomes with decision rules
- Combine multiple trees for stronger predictions

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run an example:

```bash
python 01-linear-regression/from_scratch.py
python 01-linear-regression/sklearn.py
```

## Why this repository?

The goal is not to reproduce scikit-learn. It is to make the mechanics understandable before using the library.

LinkedIn: ML Book Notes series
