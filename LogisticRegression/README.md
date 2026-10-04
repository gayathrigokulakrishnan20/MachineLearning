# Logistic Regression

**Question:** Can we predict a class using probability?

Core flow:

`X + weights + bias → z → sigmoid → probability → class`

Sigmoid:

`σ(z) = 1 / (1 + e^-z)`

The from-scratch version implements the sigmoid, log loss and gradient updates.
