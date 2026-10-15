"""Examples of common SciPy library modules.

Install dependencies with: pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, linalg, optimize, stats


def objective(x):
    """A smooth function whose minimum is at x = 0.7."""
    return np.exp((x - 0.7) ** 2)


def square(x):
    return x**2


def main():
    # scipy.optimize: find the minimum of a scalar function.
    minimum = optimize.minimize_scalar(objective)
    print("Optimization")
    print(f"  Minimum x: {minimum.x:.6f}")
    print(f"  Function value: {minimum.fun:.6f}")

    # scipy.integrate: numerically integrate x**2 from 0 to 1.
    area, estimated_error = integrate.quad(square, 0, 1)
    print("\nIntegration")
    print(f"  Integral of x^2 from 0 to 1: {area:.6f}")
    print(f"  Estimated numerical error: {estimated_error:.2e}")

    # scipy.linalg: determinant, solve a linear system, and eigenvalues.
    # This matrix is nonsingular, making it suitable for these examples.
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])
    solution = linalg.solve(matrix, rhs)
    print("\nLinear algebra")
    print(f"  Determinant: {linalg.det(matrix):.1f}")
    print(f"  Solution to A x = b: {solution}")
    print(f"  Eigenvalues: {linalg.eigvals(matrix)}")

    # scipy.stats: summary statistics and a probability calculation.
    sample = np.array([12, 14, 15, 15, 16, 18, 20])
    print("\nStatistics")
    print(f"  Mean: {stats.tmean(sample):.2f}")
    print(f"  Median: {np.median(sample):.2f}")
    print(f"  P(Z <= 1.96) for standard normal: {stats.norm.cdf(1.96):.4f}")

    # scipy.interpolate: estimate intermediate values from known data.
    x_points = np.array([0.0, 1.0, 2.0, 3.0])
    y_points = np.array([0.0, 1.0, 4.0, 9.0])
    linear_interpolator = interpolate.interp1d(x_points, y_points)
    print("\nInterpolation")
    print(f"  Estimated y at x=1.5: {float(linear_interpolator(1.5)):.2f}")

    # Optional plot of the objective function and its computed minimum.
    x = np.arange(0.0, 1.0, 0.1)
    plt.plot(x, objective(x), "o-", label="objective")
    plt.axvline(minimum.x, color="red", linestyle="--", label="minimum")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
