"""Find the best one-level decision-tree split using weighted Gini impurity."""

import numpy as np

# Features: Age, Income, Visits
X = np.array([
    [22, 25000, 8],
    [25, 30000, 7],
    [28, 35000, 6],
    [35, 40000, 2],
    [42, 45000, 3],
    [50, 60000, 1],
])

# 1 = Buy, 0 = Don't Buy
y = np.array([1, 1, 1, 0, 0, 0])

def gini(y):
    # Gini impurity is zero when every label in a group is the same.
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def weighted_gini(left_y, right_y):
    # Weight each child node's impurity by its share of all samples.
    total = len(left_y) + len(right_y)
    return (
        len(left_y) / total * gini(left_y)
        + len(right_y) / total * gini(right_y)
    )

# Track the lowest-impurity valid split across every feature and threshold.
best_score = float("inf")
best_feature = None
best_threshold = None

for feature in range(X.shape[1]):
    # Each unique observed value is a candidate split threshold.
    for threshold in np.unique(X[:, feature]):
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_y = y[left_mask]
        right_y = y[right_mask]

        # A split must place at least one sample in both child nodes.
        if len(left_y) == 0 or len(right_y) == 0:
            continue

        score = weighted_gini(left_y, right_y)

        if score < best_score:
            best_score = score
            best_feature = feature
            best_threshold = threshold

print("Best feature:", best_feature)
print("Best threshold:", best_threshold)
print("Best weighted Gini:", best_score)
