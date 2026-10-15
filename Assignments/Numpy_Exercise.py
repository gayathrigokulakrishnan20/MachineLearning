"""Common NumPy exercises and quick-reference examples."""

import numpy as np


def main() -> None:
    # 1. Create arrays and inspect their basic properties.
    values = np.array([10, 20, 30, 40, 50])
    matrix = np.arange(1, 7).reshape(2, 3)
    print("--- Array creation and properties ---")
    print("zeros:\n", np.zeros((2, 3)))
    print("ones:\n", np.ones((2, 2)))
    print("range:", np.arange(0, 10, 2))
    print("linspace:", np.linspace(0, 1, 5))
    print("matrix:\n", matrix)
    print("shape:", matrix.shape, "dimensions:", matrix.ndim, "size:", matrix.size)
    print("data type:", values.dtype)

    # 2. Select elements and filter arrays with conditions.
    print("\n--- Indexing, slicing, and filtering ---")
    print("first element:", values[0])
    print("middle values:", values[1:4])
    print("values greater than 25:", values[values > 25])

    # 3. Perform element-wise arithmetic and combine arrays.
    print("\n--- Arithmetic and broadcasting ---")
    print("values + 5:", values + 5)
    print("values * 2:", values * 2)
    print("element-wise sum:", np.array([1, 2, 3]) + np.array([4, 5, 6]))
    print("matrix plus row:\n", matrix + np.array([10, 20, 30]))

    # 4. Calculate common summary statistics.
    print("\n--- Statistics ---")
    print("sum:", np.sum(values))
    print("mean:", np.mean(values))
    print("median:", np.median(values))
    print("standard deviation:", np.std(values))
    print("minimum / maximum:", np.min(values), "/", np.max(values))

    # 5. Change array shape and transpose rows and columns.
    print("\n--- Reshape and transpose ---")
    print("flattened:", matrix.flatten())
    print("transpose:\n", matrix.T)

    # 6. Find unique values and their counts.
    print("\n--- Unique values and counts ---")
    print(np.unique([1, 2, 2, 3, 3, 3], return_counts=True))

    # 7. Generate repeatable random values with a fixed seed.
    print("\n--- Random numbers (fixed seed) ---")
    rng = np.random.default_rng(seed=42)
    print("integers:", rng.integers(0, 10, size=5))
    print("samples:", rng.random(3))

    # 8. Combine two lists and return their sorted unique values.
    a = [1, 2, 3, 4]
    b = [2, 3, 4, 5]
    print("\n--- Unique values after concatenation ---")
    print(np.unique(np.concatenate((a, b), axis=0)))

    # Shared arrays for the concatenation and stacking exercises.
    c = np.array([[1, 2, 3], [3, 4, 5]])
    d = np.array([[4, 5, 6], [7, 8, 9]])

    # 9. Concatenate arrays horizontally (along columns).
    print("\n--- Concatenate along axis=1 ---")
    print(np.concatenate((c, d), axis=1))

    # 10. Stack arrays horizontally.
    print("\n--- Horizontal stack ---")
    print(np.hstack((c, d)))

    # 11. Stack arrays vertically.
    print("\n--- Vertical stack ---")
    print(np.vstack((c, d)))

    # 12. Arrange arrays as columns.
    print("\n--- Column stack ---")
    print(np.column_stack((c, d)))


if __name__ == "__main__":
    main()
